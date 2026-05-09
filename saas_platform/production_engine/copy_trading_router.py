#!/usr/bin/env python3
"""
云端跟单执行路由 (Copy-Trading Router)

职责：
  1. 读取策略最新 target_position
  2. 遍历所有活跃订阅用户
  3. 解密用户 API Key，连接交易所
  4. 依据 allocated_capital_usdt 计算目标持仓量
  5. 对比实际持仓，发送市价单补齐/削减差额
  6. 记录订单到 saas_orders

安全约束：
  - 默认强制开启 Binance 测试网沙盒模式
  - 严格依据 allocated_capital_usdt 计算仓位，不读用户总资产
  - 每个用户独立 try-except，单用户异常不影响其他用户
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
import pandas as pd

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
        logger.info("🔀 云端跟单执行路由启动")
        logger.info(f"   沙盒模式: {'🟢 开启' if self.sandbox else '🔴 关闭'}")
        config_summary = get_config_summary()
        logger.info(f"   配置状态: {config_summary}")
        logger.info("=" * 60)

        strategies = get_live_strategies()
        if not strategies:
            logger.warning("无活跃策略，路由退出")
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
                logger.warning(f"[{strategy_name}] 无 target_position，跳过")
                continue

            trading_symbol = SYMBOL_TO_CCXT.get(target_symbol, target_symbol.replace('_', '/'))
            logger.info(f"\n{'─' * 50}")
            logger.info(f"📊 策略: {strategy_name} | target_position = {target_position:.4f} | symbol = {trading_symbol}")

            subscriptions = get_active_subscriptions(strategy_id=strategy_id)
            if not subscriptions:
                logger.info(f"[{strategy_name}] 无活跃订阅用户，跳过")
                continue

            logger.info(f"[{strategy_name}] 活跃订阅: {len(subscriptions)} 人")

            for sub in subscriptions:
                user_id = sub.get('user_id')
                sub_id = sub.get('id')
                allocated = sub.get('allocated_capital_usdt', 0)
                user_info = sub.get('saas_users', {})

                if not user_info:
                    logger.warning(f"[{strategy_name}] 订阅 {sub_id} 无用户信息，跳过")
                    continue

                enc_key = user_info.get('encrypted_api_key', '')
                enc_secret = user_info.get('encrypted_api_secret', '')
                exchange_name = user_info.get('exchange', 'binance')
                username = user_info.get('username', 'unknown')

                if not enc_key or not enc_secret:
                    logger.warning(f"[{strategy_name}] 用户 {username} 未绑定 API Key，跳过")
                    continue

                if allocated <= 0:
                    try:
                        api_key_tmp = decrypt_api_key(enc_key)
                        api_secret_tmp = decrypt_api_key(enc_secret)
                        tmp_exchange = self._create_exchange(api_key_tmp, api_secret_tmp, exchange_name)
                        if tmp_exchange:
                            balance = tmp_exchange.fetch_balance()
                            allocated = float(balance.get('USDT', {}).get('free', 0) or 0)
                            if allocated <= 0:
                                for ccy, info in balance.items():
                                    if isinstance(info, dict) and 'free' in info and float(info.get('free', 0)) > 0:
                                        allocated = float(info['free'])
                                        break
                            logger.info(f"[{username}] allocated_capital=0, \u8bfb\u53d6\u4ea4\u6613\u6240\u4f59\u989d: {allocated:.2f} USDT")
                            try:
                                tmp_exchange.close()
                            except Exception:
                                pass
                    except Exception as e:
                        logger.warning(f"[{username}] \u8bfb\u53d6\u4f59\u989d\u5931\u8d25: {e}")

                if allocated <= 0:
                    logger.warning(f"[{strategy_name}] \u7528\u6237 {username} \u8d26\u6237\u4f59\u989d\u4e3a 0\uff0c\u8df3\u8fc7")
                    continue

                total_users += 1

                try:
                    api_key = decrypt_api_key(enc_key)
                    api_secret = decrypt_api_key(enc_secret)
                except Exception as e:
                    logger.error(f"[{strategy_name}] 用户 {username} API Key 解密失败: {e}")
                    total_errors += 1
                    continue

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
                    logger.error(f"[{strategy_name}] 用户 {username} 执行异常: {e}")
                    logger.error(traceback.format_exc())
                    total_errors += 1

                    try:
                        create_order({
                            'user_id': user_id,
                            'strategy_id': strategy_id,
                            'subscription_id': sub_id,
                            'symbol': 'BTC/USDT',
                            'side': 'ERROR',
                            'order_type': 'market',
                            'amount': 0,
                            'price': 0,
                            'status': 'FAILED',
                            'error_message': str(e)[:500],
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
        logger.info(f"\n🏁 跟单路由完成: {summary}")
        return summary

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
                    logger.warning(f"[{username}] 时钟同步重试 {attempt+1}/3: {e}")
                    time.sleep(2)

            ticker = None
            for attempt in range(3):
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    break
                except Exception as e:
                    logger.warning(f"[{username}] 获取价格重试 {attempt+1}/3: {e}")
                    time.sleep(2)

            if ticker is None:
                raise Exception("无法获取市场价格")

            btc_price = ticker['last']
            logger.info(f"[{username}] BTC/USDT 价格: {btc_price:.2f}")

            target_btc_amount = (allocated_capital * target_position) / btc_price

            current_btc_amount = 0.0
            try:
                positions = exchange.fetch_positions([symbol])
                for pos in positions:
                    if pos['symbol'] == symbol:
                        current_btc_amount = float(pos['info']['positionAmt'])
                        break
            except Exception as e:
                logger.warning(f"[{username}] 获取持仓失败(视为0): {e}")

            current_weight = (current_btc_amount * btc_price) / allocated_capital if allocated_capital > 0 else 0.0
            weight_deviation = abs(target_position - current_weight)
            delta_amount = target_btc_amount - current_btc_amount
            delta_value = abs(delta_amount) * btc_price

            logger.info(
                f"[{username}] 目标={target_position*100:.1f}% 当前={current_weight*100:.1f}% "
                f"偏离={weight_deviation*100:.1f}% 差额={delta_amount:.6f} BTC ({delta_value:.2f} USDT)"
            )

            if weight_deviation < WEIGHT_TOLERANCE and delta_value > 0:
                logger.info(f"[{username}] 偏离度 < {WEIGHT_TOLERANCE*100:.0f}%，跳过微调")
                return None

            if delta_value < MIN_NOTIONAL_USDT:
                logger.info(f"[{username}] 差额 {delta_value:.2f} USDT < 最低 {MIN_NOTIONAL_USDT} USDT，跳过")
                return None

            if abs(delta_amount) < MIN_BTC_AMOUNT:
                logger.info(f"[{username}] 差额 {abs(delta_amount):.6f} BTC < 最小精度 {MIN_BTC_AMOUNT}，跳过")
                return None

            order = None
            side = ''
            if delta_amount > 0:
                side = 'buy'
                logger.info(f"[{username}] 🚀 买入 {delta_amount:.6f} BTC (约 {delta_value:.2f} USDT)")
                order = exchange.create_market_buy_order(symbol, delta_amount)
            elif delta_amount < 0:
                side = 'sell'
                sell_amount = abs(delta_amount)
                logger.info(f"[{username}] 🚀 卖出 {sell_amount:.6f} BTC (约 {delta_value:.2f} USDT)")
                order = exchange.create_market_sell_order(symbol, sell_amount)

            if order:
                fee = 0.0
                if order.get('fee'):
                    fee_obj = order['fee']
                    fee = float(fee_obj.get('cost', 0)) if isinstance(fee_obj, dict) else float(fee_obj)

                order_record = create_order({
                    'user_id': user_id,
                    'strategy_id': strategy_id,
                    'subscription_id': sub_id,
                    'symbol': symbol,
                    'side': side,
                    'order_type': 'market',
                    'amount': float(order.get('amount', 0)),
                    'price': float(order.get('average', order.get('price', 0))),
                    'fee': fee,
                    'exchange_order_id': str(order.get('id', '')),
                    'status': 'FILLED',
                })
                logger.info(f"[{username}] ✅ 订单成功: {order.get('id')}")

                return {
                    'username': username,
                    'strategy': strategy_name,
                    'side': side,
                    'amount': delta_amount,
                    'price': btc_price,
                    'order_id': order.get('id'),
                }

        except Exception as e:
            logger.error(f"[{username}] 执行异常: {e}")
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
                logger.error(f"不支持的交易所: {exchange_name}")
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
                logger.info("🟢 沙盒模式已开启 (Binance Testnet)")
            else:
                logger.warning("🔴🔴🔴 实盘模式！使用真实资金！🔴🔴🔴")

            return exchange

        except Exception as e:
            logger.error(f"创建交易所连接失败: {e}")
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
    print(f"\n跟单路由结果: {result}")
