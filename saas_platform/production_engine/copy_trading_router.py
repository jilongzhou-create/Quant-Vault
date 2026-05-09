#!/usr/bin/env python3
"""
云端跟单执行路由 (Copy-Trading Router)

逻辑对齐 trading_engine/execution_agent.py:
  - 使用 total equity (含已用保证金) 计算目标仓位
  - 实盘强制 1x 杠杆
  - 三道拦截防线: 偏离度 < 3% / 金额 < 11 USDT / 数量 < 0.001 BTC
  - 禁用 fetchCurrencies 避免超时
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
MIN_NOTIONAL = 11.0
WEIGHT_THRESHOLD = 0.03
MIN_BTC_AMOUNT = 0.001

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

        try:
            result = self._execute_for_user(
                username=username,
                api_key=api_key,
                api_secret=api_secret,
                exchange_name=exchange_name,
                strategy_id=strategy_id,
                strategy_name=strat.get('name', 'N/A'),
                target_position=target_position,
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

            if not self.sandbox:
                try:
                    exchange.set_leverage(1, symbol)
                    logger.info(f"[{username}] \U0001f512 \u5df2\u5f3a\u5236\u8bbe\u7f6e {symbol} \u6760\u6746\u4e3a 1x")
                except Exception as e:
                    logger.warning(f"[{username}] \u8bbe\u7f6e\u6760\u6746\u5931\u8d25(\u53ef\u80fd\u5df2\u662f1x): {e}")

            balance = None
            for attempt in range(3):
                try:
                    balance = exchange.fetch_balance()
                    break
                except Exception as e:
                    logger.warning(f"[{username}] \u83b7\u53d6\u4f59\u989d\u91cd\u8bd5 {attempt+1}/3: {e}")
                    time.sleep(3)

            if balance is None:
                raise Exception("\u65e0\u6cd5\u8fde\u63a5\u4ea4\u6613\u6240\u83b7\u53d6\u4f59\u989d\uff0c\u4e3a\u4fdd\u8bc1\u8d44\u91d1\u5b89\u5168\uff0c\u7ec8\u6b62\u6267\u884c\uff01")

            total_equity = float(balance.get('USDT', {}).get('total', 0) or 0)
            free_balance = float(balance.get('USDT', {}).get('free', 0) or 0)
            logger.info(f"[{username}] \u5408\u7ea6\u8d26\u6237\u603b\u6743\u76ca: {total_equity:.2f} USDT (\u53ef\u7528: {free_balance:.2f})")

            if total_equity <= 0:
                logger.warning(f"[{username}] \u8d26\u6237\u603b\u6743\u76ca\u4e3a 0\uff0c\u8df3\u8fc7")
                return None

            current_btc_amount = 0.0
            for attempt in range(3):
                try:
                    positions = exchange.fetch_positions([symbol])
                    for pos in positions:
                        if pos['symbol'] == symbol:
                            current_btc_amount = float(pos['info']['positionAmt'])
                            break
                    break
                except Exception as e:
                    logger.warning(f"[{username}] \u83b7\u53d6\u6301\u4ed3\u91cd\u8bd5 {attempt+1}/3: {e}")
                    time.sleep(2)

            logger.info(f"[{username}] \u5f53\u524d\u6301\u4ed3: {current_btc_amount:.6f} {symbol.split('/')[0]}")

            ticker = None
            for attempt in range(3):
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    break
                except Exception as e:
                    logger.warning(f"[{username}] \u83b7\u53d6\u4ef7\u683c\u91cd\u8bd5 {attempt+1}/3: {e}")
                    time.sleep(2)

            if ticker is None:
                raise Exception("\u65e0\u6cd5\u83b7\u53d6\u5e02\u573a\u6700\u65b0\u4ef7\u683c\uff0c\u7ec8\u6b62\u6267\u884c\uff01")

            current_price = ticker['last']
            logger.info(f"[{username}] {symbol} \u6700\u65b0\u4ef7\u683c: {current_price:.2f}")

            target_amount = (total_equity * target_position) / current_price
            delta_amount = target_amount - current_btc_amount
            delta_value = abs(delta_amount) * current_price
            current_weight = (current_btc_amount * current_price) / total_equity if total_equity > 0 else 0.0
            weight_deviation = abs(target_position - current_weight)

            logger.info(f"[{username}] \u76ee\u6807\u4ed3\u4f4d: {target_position*100:.1f}% | \u5f53\u524d\u4ed3\u4f4d: {current_weight*100:.1f}%")
            logger.info(f"[{username}] \u504f\u79bb\u5ea6: {weight_deviation*100:.1f}% | \u8c03\u4ed3\u5dee\u989d: {delta_amount:.6f} ({delta_value:.2f} USDT)")

            balance_before = total_equity
            position_before = current_btc_amount

            if weight_deviation < WEIGHT_THRESHOLD and delta_value > 0:
                logger.info(f"[{username}] \U0001f6d1 \u4ed3\u4f4d\u504f\u79bb\u5ea6 {weight_deviation*100:.2f}% < \u9608\u503c {WEIGHT_THRESHOLD*100:.0f}%\uff0c\u5ffd\u7565\u5fae\u8c03\uff0c\u907f\u514d\u624b\u7eed\u8d39\u78e8\u635f")
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

            if delta_value < MIN_NOTIONAL:
                logger.info(f"[{username}] \U0001f6d1 \u8c03\u4ed3\u91d1\u989d {delta_value:.2f} USDT < \u5e01\u5b89\u6700\u5c0f {MIN_NOTIONAL} USDT\uff0c\u4e0d\u6267\u884c")
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
                    'error_message': f'Below minimum notional ({delta_value:.2f} < {MIN_NOTIONAL} USDT)',
                })
                return None

            if abs(delta_amount) < MIN_BTC_AMOUNT:
                logger.info(f"[{username}] \U0001f6d1 \u8c03\u4ed3\u6570\u91cf {abs(delta_amount):.6f} < \u6700\u5c0f\u7cbe\u5ea6 {MIN_BTC_AMOUNT}\uff0c\u4e0d\u6267\u884c")
                return None

            order = None
            side = ''
            if delta_amount > 0:
                side = 'buy'
                logger.info(f"[{username}] \U0001f680 \u6267\u884c\u4e70\u5165(\u5e73\u7a7a/\u5f00\u591a): {delta_amount:.6f} ({delta_value:.2f} USDT)")
                order = exchange.create_market_buy_order(symbol, delta_amount)
            elif delta_amount < 0:
                side = 'sell'
                sell_amount = abs(delta_amount)
                logger.info(f"[{username}] \U0001f680 \u6267\u884c\u5356\u51fa(\u5e73\u591a/\u5f00\u7a7a): {sell_amount:.6f} ({delta_value:.2f} USDT)")
                order = exchange.create_market_sell_order(symbol, sell_amount)

            if order:
                fee = 0.0
                if order.get('fee'):
                    fee_obj = order['fee']
                    fee = float(fee_obj.get('cost', 0)) if isinstance(fee_obj, dict) else float(fee_obj)

                fill_price = float(order.get('average', order.get('price', 0)))

                balance_after = balance_before
                position_after = position_before
                try:
                    bal2 = exchange.fetch_balance()
                    balance_after = float(bal2.get('USDT', {}).get('total', 0) or 0)
                except Exception:
                    balance_after = balance_before

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
            exchange.options['fetchCurrencies'] = False

            if self.sandbox:
                exchange.set_sandbox_mode(True)
                logger.info("\U0001f7e2 \u6c99\u76d2\u6a21\u5f0f\u5df2\u5f00\u542f (Binance Testnet)")
            else:
                logger.warning("\U0001f534\U0001f534\U0001f534 \u5b9e\u76d8\u6a21\u5f0f\uff01\u4f7f\u7528\u771f\u5b9e\u8d44\u91d1\uff01\U0001f534\U0001f534\U0001f534")

            return exchange

        except Exception as e:
            logger.error(f"\u521c\u5efa\u4ea4\u6613\u6240\u8fde\u63a5\u5931\u8d25: {e}")
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
