
#!/usr/bin/env python3
"""
组合回测器 - 多策略对冲基金净值计算器
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:

    sys.path.insert(0, project_root)

import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime
from trading_engine.backtest_engine import load_historical_data, resample_data, compile_strategy
from database.db_manager import DB_PATH, get_strategy_version_by_id, SYMBOL_ASSET_MAP
from logger_setup import get_logger

logger = get_logger(__name__)

AVAILABLE_SYMBOLS = ['BTC_USDT', 'SPY', 'QQQ', 'GCUSD', 'PAXG_USDT', 'BZUSD']
SYMBOL_LABELS = {
    'BTC_USDT': 'BTC_USDT (加密货币)',
    'SPY': 'SPY (美股大盘)',
    'QQQ': 'QQQ (美股科技)',
    'GCUSD': 'GCUSD (黄金期货)',
    'PAXG_USDT': 'PAXG_USDT (黄金代币)',
    'BZUSD': 'BZUSD (布伦特原油)',
}


def select_target_symbol():
    print("\n" + "=" * 60)
    print("请选择回测标的:")
    print("=" * 60)
    for i, sym in enumerate(AVAILABLE_SYMBOLS, 1):
        label = SYMBOL_LABELS.get(sym, sym)
        print(f"  {i}. {label}")
    print("=" * 60)
    
    while True:
        choice = input("\n请输入选项 (1-{}): ".format(len(AVAILABLE_SYMBOLS))).strip()
        try:
            idx = int(choice)
            if 1 <= idx <= len(AVAILABLE_SYMBOLS):
                selected = AVAILABLE_SYMBOLS[idx - 1]
                print(f"✅ 已选择: {SYMBOL_LABELS.get(selected, selected)}")
                return selected
            else:
                print("❌ 无效选项")
        except ValueError:
            print("❌ 请输入数字")


def get_active_ensemble_strategies(target_symbol=None):
    """获取当前入池且有 best_version_id 的策略"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(strategy_directions)")
        columns = [row[1] for row in cursor.fetchall()]
        
        has_is_active = 'is_active_ensemble' in columns
        
        if not has_is_active:
            print("\n⚠️  检测到数据库缺少 is_active_ensemble 字段")
            print("正在自动更新数据库结构...")
            from database.db_manager import update_strategy_directions_table_structure
            update_strategy_directions_table_structure()
            print("数据库结构更新完成！\n")
        
        query = '''
        SELECT 
            sd.dir_id, 
            sd.name, 
            sd.best_version_id
        FROM strategy_directions sd
        WHERE sd.is_active_ensemble = 1 
        AND sd.best_version_id IS NOT NULL 
        AND sd.best_version_id != ''
        '''
        params = []
        
        if target_symbol:
            query += ' AND sd.target_symbol = ?'
            params.append(target_symbol)
        
        cursor.execute(query, params)
        
        rows = cursor.fetchall()
        conn.close()
        
        strategies = []
        for dir_id, name, best_version_id in rows:
            strategy = {
                'dir_id': dir_id,
                'name': name,
                'best_version_id': best_version_id
            }
            strategies.append(strategy)
        
        logger.info(f"获取到 {len(strategies)} 个入池策略")
        return strategies
    except Exception as e:
        logger.error(f"获取入池策略失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def generate_strategy_signals(strategy_version, df):
    """
    编译并运行策略，生成信号（使用正确的generate_signals函数方式）
    """
    try:
        code_content = strategy_version['code_content']
        params = strategy_version['params_json'] if strategy_version['params_json'] else {}
        
        strategy_func = compile_strategy(code_content)
        
        signals = strategy_func(df.copy(), params)
        
        if not isinstance(signals, pd.Series):
            logger.error("generate_signals 函数必须返回 pandas Series")
            return None
        
        signals = signals.reindex(df.index).fillna(0)
        return signals
    except Exception as e:
        logger.error(f"生成策略信号失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_performance_metrics(nav_series, daily_return_series):
    """计算组合业绩指标（修正所有计算）"""
    try:
        total_trading_days = len(nav_series)
        if total_trading_days < 2:
            return {}
        
        cumulative_return = nav_series.iloc[-1] - 1
        
        years = total_trading_days / 365.25
        if years > 0:
            annualized_return = (1 + cumulative_return) ** (1 / years) - 1
        else:
            annualized_return = 0
        
        peak = nav_series.expanding(min_periods=1).max()
        drawdown = (nav_series - peak) / peak
        max_drawdown = drawdown.min()
        
        periods_per_year = 365.25
        daily_std = daily_return_series.std()
        daily_mean = daily_return_series.mean()
        
        if daily_std > 0:
            sharpe_ratio = (daily_mean / daily_std) * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0
        
        return {
            'total_trading_days': total_trading_days,
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    except Exception as e:
        logger.error(f"计算业绩指标失败: {e}")
        return {}


def print_performance_report(strategy_count, metrics):
    """打印华丽的业绩报告"""
    print("\n" + "=" * 80)
    print("🎉 多策略对冲基金 - 回测业绩报告 🎉".center(80))
    print("=" * 80)
    print(f"参与策略数: {strategy_count}")
    print(f"总交易天数: {metrics.get('total_trading_days', 0)}")
    print("-" * 80)
    print(f"累计收益率: {metrics.get('cumulative_return', 0)*100:.2f}%")
    print(f"年化收益率: {metrics.get('annualized_return', 0)*100:.2f}%")
    print(f"最大回撤:   {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"组合夏普率: {metrics.get('sharpe_ratio', 0):.2f}")
    print("=" * 80)


def main():
    print("=" * 100)
    print("📊 组合回测器 - 多策略对冲基金净值计算器".center(100))
    print("=" * 100)
    
    target_symbol = select_target_symbol()
    target_asset = SYMBOL_ASSET_MAP.get(target_symbol, 'crypto')
    
    active_strategies = get_active_ensemble_strategies(target_symbol=target_symbol)
    
    if not active_strategies:
        print(f"\n❌ 错误：没有找到 {target_symbol} 入池且有效的策略！")
        print("请先使用 scripts/manage_ensemble.py 添加策略入池")
        return
    
    print(f"\n✅ 找到 {len(active_strategies)} 个 {target_symbol} 入池策略:")
    for i, s in enumerate(active_strategies, 1):
        print(f"  {i}. {s['name']} ({s['dir_id']})")
    
    print(f"\n⏳ 正在加载 {target_symbol} 历史数据与宏观因子...")
    df = load_historical_data(symbol=target_symbol, target_asset=target_asset)
    
    if df.empty:
        print("❌ 历史数据为空！")
        return
        
    print(f"✅ 历史数据加载成功，共 {len(df)} 条记录")
    
    print("\n⏳ 正在重采样为日线数据...")
    df_daily = resample_data(df, timeframe='1d')
    print(f"✅ 重采样完成，共 {len(df_daily)} 个交易日")
    
    if df_daily.empty:
        print("❌ 日线数据为空！")
        return
    
    print("\n⏳ 正在批量生成策略信号...")
    signals_dict = {}
    
    for strategy in active_strategies:
        print(f"\n  处理策略: {strategy['name']}")
        
        try:
            ver = get_strategy_version_by_id(strategy['best_version_id'])
            
            if ver is None:
                print(f"    ❌ 无法获取策略版本信息，跳过")
                continue
            
            signals = generate_strategy_signals(ver, df_daily)
            
            if signals is not None:
                signals_dict[strategy['dir_id']] = signals
                print(f"    ✅ 信号生成成功")
            else:
                print(f"    ❌ 信号生成失败，跳过")
        
        except Exception as e:
            logger.error(f"处理策略 {strategy['name']} 失败: {e}")
            print(f"    ❌ 处理失败: {e}")
    
    if not signals_dict:
        print("\n❌ 没有策略成功生成信号，退出")
        return
    
    print(f"\n✅ 共 {len(signals_dict)} 个策略成功生成信号")
    
    print("\n⏳ 正在聚合信号并计算净值...")
    signals_df = pd.DataFrame(signals_dict)
    
    combined_signal = signals_df.mean(axis=1)
    
    actual_position = combined_signal.shift(1).fillna(0)
    
    long_weight = actual_position
    cash_weight = 1.0 - actual_position.abs()
    
    turnover = actual_position.diff().fillna(0).abs()
    
    fee_paid = turnover * 0.0005
    
    price_return = df_daily['close'].pct_change().fillna(0)
    
    daily_return = actual_position * price_return - fee_paid
    
    nav = (1 + daily_return).cumprod()
    
    result_df = pd.DataFrame({
        'date': df_daily.index.strftime('%Y-%m-%d'),
        'btc_price': df_daily['close'],
        'combined_signal': combined_signal,
        'long_weight': long_weight,
        'cash_weight': cash_weight,
        'turnover': turnover,
        'fee_paid': fee_paid,
        'daily_return': daily_return,
        'nav': nav
    }).reset_index(drop=True)
    
    metrics = calculate_performance_metrics(nav, daily_return)
    print_performance_report(len(signals_dict), metrics)
    
    print("\n🎉 组合回测完成！")


if __name__ == "__main__":
    main()
