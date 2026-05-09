#!/usr/bin/env python3
"""
云端跟单执行路由 (Copy-Trading Router)
"""

import os
import sys
import logging
import traceback
import time
from datetime import datetime, timezone
from typing import Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ccxt

from saas_platform.saas_config import (
    PROXY_MODE,
    PROXY_URL,
    get_config_summary,
)
from saas_platform.database.supabase_client import (
    get_live_strategies,
    get_active_subscriptions,
    create_order,
    update_order_status,
    get_client,
    get_user_by_id,
    get_user_subscriptions,
)
from saas_platform.web_frontend.crypto_utils import decrypt_api_key

logger = logging.getLogger('saas_platform.production_engine.copy_trading_router')

SANDBOX_MODE = True
MIN_NOTIONAL_USDT = 11.0
WEIGHT_TOLERANCE = 0.03
MIN_BTC_AMOUNT = 0.001
MAX_SLIPPAGE_PCT = 0.005

SYMBOL_TO_CCXT = {
    'BTC_USDT': 'BTC/USDT',
    'ETH_USDT': 'ETH/USDT',
    'SPY': 'SPY',
    'QQQ': 'QQQ',
    'GCUSD': 'XAUUSD',
    'BZUSD': 'BZUSD',
}


class CopyTradingRouter:
    def __init__(self, sandbox: bool = SANDBOX_MODE):
        self.sandbox = sandbox
        self.results = []

    def run(self) -> dict:
        logger.info("=" * 60)
        logger.info("\U0001f500 \u4e91\u7aef\u8ddf\u5355\u6267\u884c\u8def\u7531\u542f\u52a8")
        logger.info(f"   \u6c99\u76d2\u6a21\u5f0f: {'\U0001f7e2 \u5f00\u542f' if self.sandbox else '\U0001f534 \u5173\u95ed'}")
        config_summary = get_config_summary()
        logger.info(f"   \u914d\u7f6e\u72b6\u6001: {config_summary}")
        logger.info("=" * 60)

        strategies = get_live_strategies()
        if not strategies:
            logger.warning("\u65e0\u6d3b\u8dc3\u7b56\u7565\uff0c\u8def\u7531\u9000\u51fa")
            return {'total_strategies': 0, 'total_users': 0, 'total_orders': 0, 'errors': 0}

        total_orders = 0
        total_errors = 0
        total_users = 0

        for strat in strategies:
            strategy_id = strat['id']
            strategy_name = strat.get('name', 'N/A')
            target_position = strat.get('current_target_position')
            target_symbol = strat.get('target_symbol', 'BTC_USDT')

            if target_position is None:
                logger.warning(f"[{strategy_name}] \u65e0 target_position\uff0c\u8df3\u8fc7")
                continue

            trading_symbol = SYMBOL_TO_CCXT.get(target_symbol, target_symbol.replace('_', '/'))
            logger.info(f"\n{'─' * 50}")
            logger.info(f"\U0001f4ca \u7b56\u7565: {strategy_name} | target_position = {target_position:.4f} | symbol = {trading_symbol}")

            subscriptions = get_active_subscriptions(strategy_id=strategy_id)
            if not subscriptions:
                logger.info(f"[{strategy_name}] \u65e0\u6d3b\u8dc3\u8ba2\u9605\u7528\u6237\uff0c\u8df3\u8fc7")
                continue

            logger.info(f"[{strategy_name}] \u6d3b\u8dc3\u8ba2\u9605: {len(subscriptions)} \u4eba")

            for sub in subscriptions:
                user_id = sub.get('user_id')
                sub_id = sub.get('id')
                allocated = sub.get('allocated_capital_usdt', 0)
                user_info = sub.get('saas_users', {})

                if not user_info:
                    logger.warning(f"[{strategy_name}] \u8ba2\u9605 {sub_id} \u65e0\u7528\u6237\u4fe1\u606f\uff0c\u8df3\u8fc7")
                    continue

                enc_key = user_info.get('encrypted_api_key', '')
                enc_secret = user_info.get('encrypted_api_secret', '')
                exchange_name = user_info.get('exchange', 'binance')
                username = user_info.get('username', 'unknown')

                if not enc_key or not enc_secret:
                    logger.warning(f"[{strategy_name}] \u7528\u6237 {username} \u672a\u7ed1\u5b9a API Key\uff0c\u8df3\u8fc7")
                    continue

                try:
                    api_key = decrypt_api_key(enc_key)
                    api_secret = decrypt_api_key(enc_secret)
                except Exception as e:
                    logger.error(f"[{strategy_name}] \u7528\u6237 {username} API Key \u89e3\u5bc6\u5931\u8d25: {e}")
                    total_errors += 1
                    continue

                total_users += 1

                try:
                    order_result = self._execute_for_user(
                        username=username,
                        api_key=api_key,
                        api_secret=api_secret,
                        exchange_name=exchange_name,
                        strategy_id=strategy_id,
                        strategy_name=strategy_name,
                        target_position=target_position,
                        allocated_capital=allocated,
                        sub_id=sub_id,
                        user_id=user_id,
                        trading_symbol=trading_symbol,
                    )
                    if order_result:
                        total_orders += 1
                        self.results.append(order_result)

                except Exception as e:
                    logger.error(f"[{strategy_name}] \u7528\u6237 {username} \u6267\u884c\u5f02\u5e38: {e}")
                    logger.error(traceback.format_exc())
                    total_errors += 1

                    try:
                        create_order({
                            'user_id': user_id,
                            'strategy_id': strategy_id,
                            'subscription_id': sub_id,
                            'symbol': trading_symbol,
                            'side': 'ERROR',
                            'order_type': 'market',
                            'amount': 0,
                            'price': 0,
                            'status': 'FAILED',
                            'error_message': str(e)[:500],
                            'is_sandbox': self.sandbox,
                        })
                    except Exception:
                        pass

                    continue

        summary = {
            'total_strategies': len(strategies),
            'total_users': total_users,
            'total_orders': total_orders,
            'errors': total_errors,
        }
        logger.info(f"\n\U0001f3c1 \u8ddf\u5355\u8def\u7531\u5b8c\u6210: {summary}")
        return summary

    def execute_single(self, user_id: str, strategy_id: str) -> dict:
        user = get_user_by_id(user_id)
        if not user:
            return {'success': False, 'error': 'User not found'}

        enc_key = user.get('encrypted_api_key', '')
        enc_secret = user.get('encrypted_api_secret', '')
        exchange_name = user.get('exchange', 'binance')
        username = user.get('username', 'unknown')

        if not enc_key or not enc_secret:
            return {'success': False, 'error': 'API Key not bound'}

        try:
            api_key = decrypt_api_key(enc_key)
            api_secret = decrypt_api_key(enc_secret)
        except Exception as e:
            return {'success': False, 'error': f'API Key decrypt failed: {e}'}

        subs = [x for x in get_user_subscriptions(user_id) if x.get('is_active')]
        sub = next((x for x in subs if x['strategy_id'] == strategy_id), None)
        if not sub:
            return {'success': False, 'error': 'Not subscribed to this strategy'}

        strategies = get_live_strategies()
        strat = next((x for x in strategies if x['id'] == strategy_id), None)
        if not strat:
            return {'success': False, 'error': 'Strategy not found'}

        target_position = strat.get('current_target_position')
        if target_position is None:
            return {'success': False, 'error': 'Strategy has no target position'}

        target_symbol = strat.get('target_symbol', 'BTC_USDT')
        trading_symbol = SYMBOL_TO_CCXT.get(target_symbol, target_symbol.replace('_', '/'))
        allocated = sub.get('allocated_capital_usdt', 0)

        try:
            result = self._execute_for_user(
                username=username,
                api_key=api_key,
                api_secret=api_secret,
                exchange_name=exchange_name,
                strategy_id=strategy_id,
                strategy_name=strat.get('name', 'N/A'),
                target_position=target_position,
                allocated_capital=allocated,
                sub_id=sub['id'],
                user_id=user_id,
                trading_symbol=trading_symbol,
            )
            if result:
                return {'success': True, 'order': result}
            else:
                return {'success': True, 'order': None, 'message': 'No trade needed (position within tolerance)'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _execute_for_user(
        self,
        username: str,
        api_key: str,
        api_secret: str,
        exchange_name: str,
        strategy_id: str,
        strategy_name: str,
        target_position: float,
        allocated_capital: float,
        sub_id: int,
        user_id: str,
        trading_symbol: str = 'BTC/USDT',
    ) -> Optional[dict]:
        exchange = self._create_exchange(api_key, api_secret, exchange_name)
        if exchange is None:
            return None

        try:
            symbol = trading_symbol

            for attempt in range(3):
                try:
                    exchange.load_time_difference()
                    break
                except Exception as e:
                    logger.warning(f"[{username}] \u65f6\u949f\u540c\u6b65\u91cd\u8bd5 {attempt+1}/3: {e}")
                    time.sleep(2)

            balance_before = 0.0
            try:
                bal = exchange.fetch_balance()
                balance_before = float(bal.get('USDT', {}).get('free', 0) or 0)
                if balance_before <= 0:
                    for ccy, info in bal.items():
                        if isinstance(info, dict) and 'free' in info and float(info.get('free', 0)) > 0:
                            balance_before = float(info['free'])
                            break
            except Exception as e:
                logger.warning(f"[{username}] \u83b7\u53d6\u4f59\u989d\u5931\u8d25: {e}")

            if allocated_capital <= 0:
                allocated_capital = balance_before
                logger.info(f"[{username}] allocated_capital=0, \u4f7f\u7528\u4ea4\u6613\u6240\u4f59\u989d: {allocated_capital:.2f} USDT")

            if allocated_capital <= 0:
                logger.warning(f"[{username}] \u8d26\u6237\u4f59\u989d\u4e3a 0\uff0c\u8df3\u8fc7")
                return None

            ticker = None
            for attempt in range(3):
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    break
                except Exception as e:
                    logger.warning(f"[{username}] \u83b7\u53d6\u4ef7\u683c\u91cd\u8bd5 {attempt+1}/3: {e}")
                    time.sleep(2)

            if ticker is None:
                raise Exception("\u65e0\u6cd5\u83b7\u53d6\u5e02\u573a\u4ef7\u683c")

            current_price = ticker['last']
            logger.info(f"[{username}] {symbol} \u4ef7\u683c: {current_price:.2f}")

            target_amount = (allocated_capital * target_position) / current_price

            position_before = 0.0
            try:
                positions = exchange.fetch_positions([symbol])
                for pos in positions:
                    if pos['symbol'] == symbol:
                        position_before = float(pos['info']['positionAmt'])
                        break
            except Exception as e:
                logger.warning(f"[{username}] \u83b7\u53d6\u6301\u4ed3\u5931\u8d25(\u89c6\u4e3a0): {e}")

            current_weight = (position_before * current_price) / allocated_capital if allocated_capital > 0 else 0.0
            weight_deviation = abs(target_position - current_weight)
            delta_amount = target_amount - position_before
            delta_value = abs(delta_amount) * current_price

            logger.info(
                f"[{username}] \u76ee\u6807={target_position*100:.1f}% \u5f53\u524d={current_weight*100:.1f}% "
                f"\u504f\u79bb={weight_deviation*100:.1f}% \u5dee\u989d={delta_amount:.6f} ({delta_value:.2f} USDT)"
            )

            if weight_deviation < WEIGHT_TOLERANCE and delta_value > 0:
                logger.info(f"[{username}] \u504f\u79bb\u5ea6 < {WEIGHT_TOLERANCE*100:.0f}%\uff0c\u8df3\u8fc7\u5fae\u8c03")
                create_order({
                    'user_id': user_id,
                    'strategy_id': strategy_id,
                    'subscription_id': sub_id,
                    'symbol': symbol,
                    'side': 'SKIP',
                    'order_type': 'market',
                    'amount': 0,
                    'price': current_price,
                    'status': 'SKIPPED',
                    'target_position': target_position,
                    'balance_before': balance_before,
                    'balance_after': balance_before,
                    'position_before': position_before,
                    'position_after': position_before,
                    'notional_value': delta_value,
                    'is_sandbox': self.sandbox,
                    'error_message': f'Position within tolerance (deviation {weight_deviation*100:.1f}%)',
                })
                return None

            if delta_value < MIN_NOTIONAL_USDT:
                logger.info(f"[{username}] \u5dee\u989d {delta_value:.2f} USDT < \u6700\u4f4e {MIN_NOTIONAL_USDT} USDT\uff0c\u8df3\u8fc7")
                create_order({
                    'user_id': user_id,
                    'strategy_id': strategy_id,
                    'subscription_id': sub_id,
                    'symbol': symbol,
                    'side': 'SKIP',
                    'order_type': 'market',
                    'amount': 0,
                    'price': current_price,
                    'status': 'SKIPPED',
                    'target_position': target_position,
                    'balance_before': balance_before,
                    'balance_after': balance_before,
                    'position_before': position_before,
                    'position_after': position_before,
                    'notional_value': delta_value,
                    'is_sandbox': self.sandbox,
                    'error_message': f'Below minimum notional ({delta_value:.2f} < {MIN_NOTIONAL_USDT} USDT)',
                })
                return None

            if abs(delta_amount) < MIN_BTC_AMOUNT:
                logger.info(f"[{username}] \u5dee\u989d {abs(delta_amount):.6f} < \u6700\u5c0f\u7cbe\u5ea6 {MIN_BTC_AMOUNT}\uff0c\u8df3\u8fc7")
                return None

            order = None
            side = ''
            if delta_amount > 0:
                side = 'buy'
                logger.info(f"[{username}] \U0001f680 \u4e70\u5165 {delta_amount:.6f} ({delta_value:.2f} USDT)")
                order = exchange.create_market_buy_order(symbol, delta_amount)
            elif delta_amount < 0:
                side = 'sell'
                sell_amount = abs(delta_amount)
                logger.info(f"[{username}] \U0001f680 \u5356\u51fa {sell_amount:.6f} ({delta_value:.2f} USDT)")
                order = exchange.create_market_sell_order(symbol, sell_amount)

            if order:
                fee = 0.0
                if order.get('fee'):
                    fee_obj = order['fee']
                    fee = float(fee_obj.get('cost', 0)) if isinstance(fee_obj, dict) else float(fee_obj)

                fill_price = float(order.get('average', order.get('price', 0)))

                balance_after = 0.0
                position_after = position_before
                try:
                    bal2 = exchange.fetch_balance()
                    balance_after = float(bal2.get('USDT', {}).get('free', 0) or 0)
                    if balance_after <= 0:
                        for ccy, info in bal2.items():
                            if isinstance(info, dict) and 'free' in info and float(info.get('free', 0)) > 0:
                                balance_after = float(info['free'])
                                break
                except Exception:
                    balance_after = balance_before - delta_value if side == 'buy' else balance_before + delta_value

                try:
                    positions2 = exchange.fetch_positions([symbol])
                    for pos in positions2:
                        if pos['symbol'] == symbol:
                            position_after = float(pos['info']['positionAmt'])
                            break
                except Exception:
                    position_after = position_before + delta_amount

                create_order({
                    'user_id': user_id,
                    'strategy_id': strategy_id,
                    'subscription_id': sub_id,
                    'symbol': symbol,
                    'side': side,
                    'order_type': 'market',
                    'amount': float(order.get('amount', 0)),
                    'price': fill_price,
                    'fee': fee,
                    'exchange_order_id': str(order.get('id', '')),
                    'status': 'FILLED',
                    'target_position': target_position,
                    'balance_before': balance_before,
                    'balance_after': balance_after,
                    'position_before': position_before,
                    'position_after': position_after,
                    'notional_value': delta_value,
                    'is_sandbox': self.sandbox,
                })
                logger.info(f"[{username}] \u2705 \u8ba2\u5355\u6210\u529f: {order.get('id')}")

                return {
                    'username': username,
                    'strategy': strategy_name,
                    'side': side,
                    'amount': delta_amount,
                    'price': fill_price,
                    'order_id': order.get('id'),
                    'balance_before': balance_before,
                    'balance_after': balance_after,
                    'position_before': position_before,
                    'position_after': position_after,
                    'fee': fee,
                    'is_sandbox': self.sandbox,
                }

        except Exception as e:
            logger.error(f"[{username}] \u6267\u884c\u5f02\u5e38: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            try:
                exchange.close()
            except Exception:
                pass

        return None

    def _create_exchange(self, api_key: str, api_secret: str, exchange_name: str = 'binance') -> Optional[ccxt.Exchange]:
        try:
            exchange_class = getattr(ccxt, exchange_name, None)
            if exchange_class is None:
                logger.error(f"\u4e0d\u652f\u6301\u7684\u4ea4\u6613\u6240: {exchange_name}")
                return None

            options = {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,
                },
            }

            if PROXY_MODE == 'proxy' and PROXY_URL:
                options['proxies'] = {
                    'http': PROXY_URL,
                    'https': PROXY_URL,
                }

            exchange = exchange_class(options)

            if self.sandbox:
                exchange.set_sandbox_mode(True)
                logger.info("\U0001f7e2 \u6c99\u76d2\u6a21\u5f0f\u5df2\u5f00\u542f (Binance Testnet)")
            else:
                logger.warning("\U0001f534\U0001f534\U0001f534 \u5b9e\u76d8\u6a21\u5f0f\uff01\u4f7f\u7528\u771f\u5b9e\u8d44\u91d1\uff01\U0001f534\U0001f534\U0001f534")

            return exchange

        except Exception as e:
            logger.error(f"\u521b\u5efa\u4ea4\u6613\u6240\u8fde\u63a5\u5931\u8d25: {e}")
            return None


def run_copy_trading(sandbox: bool = SANDBOX_MODE) -> dict:
    router = CopyTradingRouter(sandbox=sandbox)
    return router.run()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    )
    result = run_copy_trading()
    print(f"\n\u8ddf\u5355\u8def\u7531\u7ed3\u679c: {result}")
