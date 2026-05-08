#!/usr/bin/env python3
"""
全自动实盘/模拟盘执行大脑 - Daemon 守护进程
每天定时执行数据同步与交易
"""

import sys
import os
import subprocess
import time
from datetime import datetime, date
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:

    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import ccxt
from trading_engine.backtest_engine import load_daily_data_directly, compile_strategy
from database.db_manager import (
    DB_PATH,
    get_portfolios_by_status,
    get_portfolio_strategies_with_code,
    save_portfolio_records,
    save_exchange_order
)
from logger_setup import get_logger

# 加载环境变量
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

# 读取真实 API Keys
BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY')

# 读取实盘开关 (默认为 False)
LIVE_TRADING = os.environ.get('LIVE_TRADING', 'False').lower() == 'true'

# 读取做空开关 (默认为 False)
ENABLE_SHORTING = os.environ.get('ENABLE_SHORTING', 'False').lower() == 'true'

# 打印调试信息
print(f"===== 调试信息 =====")
print(f".env 路径: {env_path}")
print(f"BINANCE_API_KEY: {'***已配置***' if BINANCE_API_KEY else '未配置'}")
print(f"BINANCE_SECRET_KEY: {'***已配置***' if BINANCE_SECRET_KEY else '未配置'}")
print(f"LIVE_TRADING: {LIVE_TRADING}")
print(f"====================")

logger = get_logger(__name__)

# 注意：使用前请确保安装 schedule 库
# pip install schedule
try:
    import schedule
except ImportError:
    logger.warning("schedule 库未安装，请运行: pip install schedule")
    schedule = None


def run_data_sync():
    """
    Step 1: 执行每日全量数据同步
    """
    logger.info("⏳ [Step 1] 开始执行每日全量数据同步...")
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_pipeline', 'sync_crypto_data.py')
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ 数据同步圆满完成！")
            return True
        else:
            logger.error(f"❌ 数据同步失败，退出今日交易: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ 触发数据同步异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_latest_signal(strategies):
    """
    获取组合中所有策略的最新信号，等权平均
    """
    try:
        logger.info("开始获取最新组合信号")
        
        logger.info("⏳ 正在高效加载日线数据...")
        df_daily = load_daily_data_directly()
        
        if df_daily.empty:
            logger.error("❌ 日线数据加载失败或为空！")
            return 0
            
        df_daily = df_daily.reset_index()
        logger.info(f"✅ 日线数据准备完成，共 {len(df_daily)} 个交易日")
            
        signals_list = []
        
        for strategy in strategies:
            dir_id = strategy['dir_id']
            code_content = strategy['best_code']
            params_json = strategy.get('params', {})
            
            if not code_content:
                logger.warning(f"策略 {dir_id} 没有代码，跳过")
                continue
                
            try:
                strategy_func = compile_strategy(code_content)
                signals = strategy_func(df_daily.copy(), params_json)
                
                if isinstance(signals, pd.DataFrame):
                    if 'signal' in signals.columns:
                        signal_series = signals['signal']
                    elif 'signals' in signals.columns:
                        signal_series = signals['signals']
                    else:
                        logger.warning(f"策略 {dir_id} 返回 DataFrame 但无 signal 列")
                        continue
                else:
                    signal_series = pd.Series(signals)
                    
                if len(signal_series) > 0:
                    latest_signal = signal_series.iloc[-1]
                    signals_list.append(latest_signal)
                    logger.info(f"策略 {dir_id} 最新信号: {latest_signal}")
                else:
                    logger.warning(f"策略 {dir_id} 生成的信号为空")
                    
            except Exception as e:
                logger.error(f"策略 {dir_id} 执行失败: {e}")
                continue
        
        if len(signals_list) == 0:
            logger.error("没有有效的策略信号")
            return 0
            
        target_signal = np.mean(signals_list)
        if ENABLE_SHORTING:
            target_signal = np.clip(target_signal, -1.0, 1.0)
        else:
            target_signal = np.clip(target_signal, 0.0, 1.0)
        
        logger.info(f"组合最新总信号: {target_signal:.4f} (基于 {len(signals_list)} 个有效策略)")
        return target_signal
        
    except Exception as e:
        logger.error(f"获取最新信号失败: {e}")
        import traceback
        traceback.print_exc()
        return 0


def execute_trading(portfolio, target_signal):
    """
    Step 2-4: 执行交易逻辑（对齐目标仓位）
    返回: (success, result, run_phase_label)
    """
    portfolio_id = portfolio['portfolio_id']
    portfolio_status = portfolio['status']
    result = {
        'portfolio_id': portfolio_id,
        'status': portfolio_status,
        'orders': [],
        'total_equity': 0,
        'btc_price': 0,
        'usdt_free': 0,
        'btc_free': 0
    }
    
    try:
        logger.info(f"开始执行组合 {portfolio_id} ({portfolio_status}) 交易")
        
        # 调试信息：检查 API Key 是否加载
        if BINANCE_API_KEY:
            logger.info(f"✅ BINANCE_API_KEY 已加载: {BINANCE_API_KEY[:20]}...")
        else:
            logger.error("❌ BINANCE_API_KEY 未加载！")
        
        if BINANCE_SECRET_KEY:
            logger.info(f"✅ BINANCE_SECRET_KEY 已加载: {BINANCE_SECRET_KEY[:20]}...")
        else:
            logger.error("❌ BINANCE_SECRET_KEY 未加载！")
        
        exchange_options = {
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
                'recvWindow': 60000
            }
        }
        
        http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        
        if http_proxy or https_proxy:
            exchange_options['proxies'] = {}
            if http_proxy:
                exchange_options['proxies']['http'] = http_proxy
            if https_proxy:
                exchange_options['proxies']['https'] = https_proxy
        
        exchange = ccxt.binance(exchange_options)
        
        if not LIVE_TRADING:
            logger.info("🟢 当前处于模拟盘模式 (Sandbox/Paper Trading)")
            exchange.set_sandbox_mode(True)
            exchange.urls['api']['public'] = 'https://testnet.binance.vision/api/v3'
            exchange.urls['api']['private'] = 'https://testnet.binance.vision/api/v3'
            run_phase_label = 'PAPER'
        else:
            # 终极实盘警告与倒计时
            logger.warning("=" * 60)
            logger.warning("🚨 警告：当前已开启实盘模式 (LIVE_TRADING=True)！")
            logger.warning("🚨 系统即将使用真实资金进行交易！")
            logger.warning("=" * 60)
            print("如果你是误操作，请在 10 秒内按 Ctrl+C 取消...")
            for i in range(10, 0, -1):
                print(f"{i}...", end=" ", flush=True)
                time.sleep(1)
            print("\n🚀 正在连接币安真实服务器...")
            exchange.set_sandbox_mode(False)
            run_phase_label = 'LIVE'
        
        # 【安全防线 1】：强制拉取币安服务器时间，计算精准偏移量，彻底消灭 -1021 错误
        logger.info("⏳ 正在与币安服务器进行底层时钟同步...")
        time_synced = False
        for attempt in range(3):
            try:
                exchange.load_time_difference()
                time_synced = True
                logger.info(f"时钟同步完成，时间偏移: {exchange.options.get('timeDifference', '未知')} ms")
                break
            except Exception as e:
                logger.warning(f"时钟同步异常，重试 {attempt+1}/3: {e}")
                time.sleep(2)

        if not time_synced:
            logger.warning("合约端点时钟同步失败，尝试使用现货端点回退...")
            try:
                original_type = exchange.options.get('defaultType')
                exchange.options['defaultType'] = 'spot'
                exchange.load_time_difference()
                exchange.options['defaultType'] = original_type or 'future'
                time_synced = True
                logger.info(f"✅ 通过现货端点完成时钟同步，时间偏移: {exchange.options.get('timeDifference', '未知')} ms")
            except Exception as e:
                logger.warning(f"现货端点时钟同步也失败: {e}")
        
        # 禁用 fetchCurrencies 避免访问现货 API 超时
        exchange.options['fetchCurrencies'] = False
        
        # 【安全防线 2】：实盘模式强制设置 1x 杠杆 (严格无杠杆)
        if LIVE_TRADING:
            try:
                exchange.set_leverage(1, 'BTC/USDT')
                logger.info("🔒 已强制设置 BTC/USDT 杠杆为 1x (严格无杠杆)")
            except Exception as e:
                logger.warning(f"设置杠杆失败(可能已是1x): {e}")
        
        # 【安全防线 3】：获取合约账户余额，加入防 SSL 断连的安全重试机制
        logger.info("获取合约账户余额...")
        balance = None
        for attempt in range(3):
            try:
                balance = exchange.fetch_balance()
                break
            except Exception as e:
                logger.warning(f"网络/SSL异常，重试获取余额 {attempt+1}/3: {e}")
                time.sleep(3)
        
        if balance is None:
            raise Exception("无法连接交易所获取余额，为保证资金安全，终止执行！")
        
        total_equity = float(balance['USDT']['total'])
        logger.info(f"合约账户总权益: {total_equity:.2f} USDT")
        
        # 获取当前合约持仓
        logger.info("获取 BTC/USDT 合约持仓...")
        current_btc_amount = 0.0
        positions = None
        for attempt in range(3):
            try:
                positions = exchange.fetch_positions(['BTC/USDT'])
                break
            except Exception as e:
                logger.warning(f"获取持仓异常，重试 {attempt+1}/3: {e}")
                time.sleep(2)
        
        if positions:
            for pos in positions:
                if pos['symbol'] == 'BTC/USDT':
                    current_btc_amount = float(pos['info']['positionAmt'])
                    break
        
        logger.info(f"当前合约持仓: {current_btc_amount:.6f} BTC")
        
        # 获取最新价格
        logger.info("获取 BTC/USDT 最新价格...")
        ticker = None
        for attempt in range(3):
            try:
                ticker = exchange.fetch_ticker('BTC/USDT')
                break
            except Exception as e:
                logger.warning(f"获取价格异常，重试 {attempt+1}/3: {e}")
                time.sleep(2)
        
        if ticker is None:
            raise Exception("无法获取市场最新价格，终止执行！")
        
        btc_price = ticker['last']
        logger.info(f"BTC/USDT 最新价格: {btc_price:.2f}")
        
        result['total_equity'] = total_equity
        result['btc_price'] = btc_price
        result['current_btc_amount'] = current_btc_amount
        
        # 计算目标仓位与调仓差额
        target_btc_amount = (total_equity * target_signal) / btc_price
        delta_amount = target_btc_amount - current_btc_amount
        delta_value = abs(delta_amount) * btc_price
        
        current_weight = (current_btc_amount * btc_price) / total_equity if total_equity > 0 else 0.0
        
        logger.info(f"AI 目标信号 (Target Weight): {target_signal*100:.2f}%")
        logger.info(f"当前实际仓位 (Current Weight): {current_weight*100:.2f}%")
        logger.info(f"目标 BTC 数量: {target_btc_amount:.6f}")
        logger.info(f"当前 BTC 数量: {current_btc_amount:.6f}")
        logger.info(f"调仓差额: {delta_amount:.6f} BTC (约 {delta_value:.2f} USDT)")
        
        # 币安最小交易额要求
        MIN_NOTIONAL = 11.0
        
        # 【核心防御】：设定 3% 的容忍阈值 (Tolerance Band)
        WEIGHT_THRESHOLD = 0.03
        weight_deviation = abs(target_signal - current_weight)
        
        # 拦截逻辑 1：偏离度太小，忽略价格微小波动
        if weight_deviation < WEIGHT_THRESHOLD and delta_value > 0:
            logger.info(f"🛑 仓位偏离度 {weight_deviation*100:.2f}% 小于阈值 {WEIGHT_THRESHOLD*100:.0f}%，忽略价格微调，避免手续费磨损！")
            return True, result, run_phase_label
            
        # 拦截逻辑 2：绝对金额太小，达不到币安最小限制
        if delta_value < MIN_NOTIONAL:
            logger.info(f"🛑 理论调仓金额 {delta_value:.2f} USDT 小于币安最小交易额 {MIN_NOTIONAL} USDT，不执行交易！")
            return True, result, run_phase_label
        
        # BTC/USDT 合约最小下单数量 (币安规定 0.001 BTC)
        min_amount = 0.001
        
        # 拦截逻辑 3：调仓数量小于最小币种数量精度
        if abs(delta_amount) < min_amount:
            logger.info(f"🛑 理论调仓数量 {abs(delta_amount):.6f} 小于币安合约最小限制 {min_amount}，不执行交易！")
            return True, result, run_phase_label
            
        # 通过了拦截，准备执行真实交易
        if delta_amount > 0:
            logger.info(f"🚀 执行买入(平空/开多): BTC 数量={delta_amount:.6f}, 价值约={delta_value:.2f} USDT")
            
            try:
                order = exchange.create_market_buy_order('BTC/USDT', delta_amount)
                order['portfolio_id'] = portfolio_id
                result['orders'].append(order)
                logger.info(f"✅ 买入订单成功: order_id={order['id']}")
            except Exception as e:
                logger.error(f"❌ 买入订单失败: {e}")
                raise e
                
        elif delta_amount < 0:
            sell_amount = abs(delta_amount)
            logger.info(f"🚀 执行卖出(平多/开空): BTC 数量={sell_amount:.6f}, 价值约={delta_value:.2f} USDT")
            
            try:
                order = exchange.create_market_sell_order('BTC/USDT', sell_amount)
                order['portfolio_id'] = portfolio_id
                result['orders'].append(order)
                logger.info(f"✅ 卖出订单成功: order_id={order['id']}")
            except Exception as e:
                logger.error(f"❌ 卖出订单失败: {e}")
                raise e
                
        return True, result, run_phase_label
        
    except Exception as e:
        logger.error(f"执行交易失败: {e}")
        import traceback
        traceback.print_exc()
        return False, result, 'PAPER'


def save_execution_results(portfolio, execution_result, target_signal, run_phase_label):
    """
    保存执行结果到数据库
    """
    try:
        portfolio_id = portfolio['portfolio_id']
        today = date.today().strftime('%Y-%m-%d')
        
        logger.info(f"保存组合 {portfolio_id} 执行结果 (阶段: {run_phase_label})")
        
        for order in execution_result['orders']:
            try:
                save_exchange_order(order)
            except Exception as e:
                logger.error(f"保存订单失败: {e}")
                import traceback
                traceback.print_exc()
        
        total_fee = 0
        for order in execution_result['orders']:
            if 'fee' in order and order['fee']:
                fee_obj = order['fee']
                if isinstance(fee_obj, dict):
                    total_fee += float(fee_obj.get('cost', 0))
                else:
                    total_fee += float(fee_obj)
            elif 'fees' in order and len(order['fees']) > 0:
                total_fee += sum(float(f.get('cost', 0)) for f in order['fees'])
        
        record = {
            'date': today,
            'btc_price': execution_result['btc_price'],
            'combined_signal': target_signal,
            'turnover': 0,
            'fee_paid': total_fee,
            'daily_return': 0,
            'nav': execution_result['total_equity'],
            'total_equity': execution_result['total_equity']
        }
        
        df_record = pd.DataFrame([record])
        save_portfolio_records(portfolio_id, df_record, run_phase_label)
        
        logger.info(f"执行结果保存完成")
        
    except Exception as e:
        logger.error(f"保存执行结果失败: {e}")


def daily_trading_routine():
    """
    每日交易例行程序 (Step 1 - Step 4)
    """
    print("\n" + "=" * 80)
    print(f"📅 开始执行每日交易例行程序 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        if not run_data_sync():
            logger.error("数据同步失败，终止今日交易")
            return
            
        logger.info("查询正在运行的组合...")
        portfolios = get_portfolios_by_status(['PAPER', 'LIVE'])
        
        if not portfolios:
            print("\n❌ 没有找到正在运行的组合（需要将其 status 改为 PAPER 或 LIVE）")
            return
            
        if len(portfolios) > 1:
            logger.warning(f"\n⚠️ 警告：检测到 {len(portfolios)} 个处于 PAPER/LIVE 状态的组合！")
            logger.warning("由于目前使用全局单一 API 账户，多组合同时运行会导致资金互相抢夺（左右互搏）。")
            logger.warning(f"🔒 系统已触发安全锁：将仅执行第一个组合 (ID: {portfolios[0]['portfolio_id']})，强制跳过其他组合！\n")
            portfolios = [portfolios[0]]
            
        print(f"\n✅ 锁定目标组合: {portfolios[0]['name']} (ID: {portfolios[0]['portfolio_id']})")
        
        for portfolio in portfolios:
            print(f"\n{'=' * 80}")
            print(f"处理组合: {portfolio['name']} (ID: {portfolio['portfolio_id']}, 状态: {portfolio['status']})")
            print('=' * 80)
            
            try:
                strategies = get_portfolio_strategies_with_code(portfolio['portfolio_id'])
                
                if not strategies:
                    print(f"❌ 组合 {portfolio['portfolio_id']} 没有关联策略，跳过")
                    continue
                    
                print(f"✅ 加载了 {len(strategies)} 个策略")
                
                target_signal = get_latest_signal(strategies)
                
                if target_signal is None:
                    print(f"❌ 获取信号失败，跳过")
                    continue
                
                try:
                    trade_success, execution_result, run_phase_label = execute_trading(portfolio, target_signal)
                    
                    if not trade_success:
                        logger.error(f"❌ 组合 {portfolio['portfolio_id']} 交易执行失败，中止当日落库记录，保护对账单一致性！")
                        print(f"\n❌ 组合 {portfolio['portfolio_id']} 处理失败，跳过")
                        continue
                        
                except Exception as e:
                    logger.error(f"❌ 组合 {portfolio['portfolio_id']} 交易执行发生异常: {e}，中止落库！")
                    import traceback
                    traceback.print_exc()
                    continue
                
                logger.info("✅ 交易成功执行，正在将实际记录写入数据库...")
                save_execution_results(portfolio, execution_result, target_signal, run_phase_label)
                
                print(f"\n✅ 组合 {portfolio['portfolio_id']} 处理完成")
                
            except Exception as e:
                logger.error(f"处理组合 {portfolio['portfolio_id']} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("今日交易例行程序全部完成")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"每日例行程序执行失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    主函数 - 守护进程模式
    """
    print("\n" + "=" * 80)
    print("🤖 全自动实盘/模拟盘执行大脑已启动")
    print("=" * 80)
    
    if schedule is None:
        print("\n⚠️  schedule 库未安装，仅支持单次运行模式")
        choice = input("是否立即执行一次？(y/n): ").strip().lower()
        if choice == 'y':
            daily_trading_routine()
        return
    
    schedule.every().day.at("00:05").do(daily_trading_routine)
    logger.info("⏰ 已设置定时任务，每天 UTC 00:05 自动执行数据同步与交易...")
    
    choice = input("是否立即执行一次测试运行？(y/n): ").strip().lower()
    if choice == 'y':
        daily_trading_routine()
    
    print("\n💤 守护进程已启动，等待定时任务...")
    print("   按 Ctrl+C 可以停止程序\n")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n👋 程序已停止")


if __name__ == "__main__":
    main()
