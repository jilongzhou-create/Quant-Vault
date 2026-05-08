
#!/usr/bin/env python3
"""
组合回测计算器 - 基于 Portfolio ID 的标准回测引擎
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
from trading_engine.backtest_engine import load_daily_data_directly, compile_strategy
from database.db_manager import (
    DB_PATH,
    SYMBOL_ASSET_MAP,
    get_portfolios_by_status,
    get_portfolio_strategies_with_code,
    save_portfolio_records,
    update_portfolio_status,
    update_portfolio_metrics
)
from logger_setup import get_logger

logger = get_logger(__name__)


def select_portfolio():
    """
    列出 DRAFT 和 TESTED 状态的组合，让用户选择
    """
    portfolios = get_portfolios_by_status(['DRAFT', 'TESTED'])
    
    if not portfolios:
        print("\n❌ 没有找到可用的组合！")
        print("请先使用 scripts/manage_ensemble.py 创建组合")
        return None
    
    print("\n" + "=" * 90)
    print(f"{'ID':<6} {'组合名称':<30} {'标的':<12} {'状态':<12} {'创建时间':<20}")
    print("-" * 90)
    
    for p in portfolios:
        print(f"{p['portfolio_id']:<6} {p['name'][:28]:<30} {p.get('target_symbol', 'BTC_USDT'):<12} {p['status']:<12} {p['created_at'][:16]:<20}")
    
    print("=" * 90)
    
    while True:
        choice = input("\n请输入要测试的组合 ID: ").strip()
        try:
            portfolio_id = int(choice)
            selected = next((p for p in portfolios if p['portfolio_id'] == portfolio_id), None)
            if selected:
                return selected
            else:
                print("❌ 无效的组合 ID，请重新输入")
        except ValueError:
            print("❌ 请输入有效的数字 ID")


def get_strategy_sharpe_rates(strategies):
    """
    从数据库获取每个策略的夏普率
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        sharpe_rates = {}
        for strategy in strategies:
            dir_id = strategy['dir_id']
            cursor.execute('''
                SELECT sv.metric_sharpe
                FROM strategy_directions sd
                JOIN strategy_versions sv ON sd.best_version_id = sv.ver_id
                WHERE sd.dir_id = ?
                  AND sv.run_status != 'OVERFITTED'
            ''', (dir_id,))
            result = cursor.fetchone()
            if result and result[0] is not None:
                sharpe_rates[dir_id] = result[0]
            else:
                sharpe_rates[dir_id] = 0
        
        conn.close()
        return sharpe_rates
    except Exception as e:
        logger.error(f"获取策略夏普率失败: {e}")
        return {}


def select_weight_mode():
    """
    让用户选择权重模式
    """
    print("\n" + "=" * 60)
    print("请选择资金分配模式:")
    print("=" * 60)
    print("  1. 等权平均 (Equal Weight) - 经典防守型")
    print("  2. 夏普加权 (Sharpe Weighted) - 优胜劣汰型")
    print("  3. 目标满仓缩放 (Fully Invested Scaling) - 激进攻击型")
    print("  4. 风险平价 (Risk Parity) - 机构级波动率均衡")
    print("=" * 60)
    
    while True:
        choice = input("\n请输入选项 (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            mode_map = {
                '1': ('等权平均', 'equal'),
                '2': ('夏普加权', 'sharpe'),
                '3': ('目标满仓缩放', 'scaling'),
                '4': ('风险平价', 'risk_parity')
            }
            return mode_map[choice]
        else:
            print("❌ 无效选项，请重新选择 (1-4)")


def generate_strategy_signals(code_content, params, df):
    """
    编译并运行策略，生成信号（保留原有的优秀逻辑）
    """
    try:
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


def combine_signals(signals_df, mode, strategy_metrics=None, price_return=None):
    """
    根据选择的模式聚合信号
    """
    if mode == 'equal':
        # 等权平均
        combined_signal = signals_df.mean(axis=1)
        
    elif mode == 'sharpe':
        # 夏普加权
        if strategy_metrics is None:
            strategy_metrics = {}
        
        weights = {}
        total_sharpe = 0
        
        for dir_id in signals_df.columns:
            sharpe = strategy_metrics.get(dir_id, 0)
            if sharpe > 0:
                weights[dir_id] = sharpe
                total_sharpe += sharpe
            else:
                weights[dir_id] = 0
        
        # 归一化
        if total_sharpe > 0:
            for dir_id in weights:
                weights[dir_id] = weights[dir_id] / total_sharpe
        
        # 加权求和
        combined_signal = pd.Series(0, index=signals_df.index)
        for dir_id, weight in weights.items():
            if weight > 0:
                combined_signal += signals_df[dir_id] * weight
        
    elif mode == 'scaling':
        # 目标满仓缩放 (Directional Independent Scaling / 多空独立缩放)
        # 1. 算出每天信号的直接加总（合力），而不是均值
        raw_sum = signals_df.sum(axis=1)
        
        # 2. 分别找出过去历史中，多头的最大合力和空头的最大合力
        # 使用 .max() 和 .min()，并添加后备保护避免分母为0
        max_positive = raw_sum[raw_sum > 0].max() if (raw_sum > 0).any() else 1.0
        max_negative = raw_sum[raw_sum < 0].min() if (raw_sum < 0).any() else -1.0
        
        # 3. 独立缩放逻辑：正数除以最大正向极值，负数除以最大负向极值的绝对值
        def scale_signal(val):
            if val > 0:
                return val / max_positive
            elif val < 0:
                return val / abs(max_negative)
            else:
                return 0.0
                
        # 4. 应用缩放并赋值
        combined_signal = raw_sum.apply(scale_signal)
    elif mode == 'risk_parity':
        # 机构级波动率风险平价 (Inverse Volatility Weighting)
        if price_return is None:
            logger.warning("未传入 price_return，降级为等权模式")
            combined_signal = signals_df.mean(axis=1)
        else:
            weights = {}
            for col in signals_df.columns:
                strat_daily_return = signals_df[col].shift(1).fillna(0) * price_return
                # 计算标准差 (波动率)
                volatility = strat_daily_return.std()
                
                # 2. 取波动率的倒数 (波动越小，权重越大)
                # 增加 1e-6 防止除以 0 的情况（比如策略从未交易过）
                if volatility > 1e-6:
                    weights[col] = 1.0 / volatility
                else:
                    weights[col] = 0.0
                    
            # 3. 权重归一化 (使其总和为 1.0)
            total_inv_vol = sum(weights.values())
            combined_signal = pd.Series(0.0, index=signals_df.index)
            
            if total_inv_vol > 0:
                for col, w in weights.items():
                    normalized_weight = w / total_inv_vol
                    combined_signal += signals_df[col] * normalized_weight
            else:
                # 极端异常后备方案
                combined_signal = signals_df.mean(axis=1)
    else:
        combined_signal = signals_df.mean(axis=1)
    
    # 限制在 [-1, 1] 之间
    combined_signal = combined_signal.clip(-1, 1)
    return combined_signal


def calculate_performance_metrics(nav_series, daily_return_series):
    """
    计算组合业绩指标
    """
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


def print_performance_report(strategy_count, metrics, mode_name):
    """
    打印华丽的业绩报告
    """
    print("\n" + "=" * 80)
    print("🎉 多策略组合 - 回测业绩报告 🎉".center(80))
    print("=" * 80)
    print(f"权重模式: {mode_name}")
    print(f"参与策略数: {strategy_count}")
    print(f"总交易天数: {metrics.get('total_trading_days', 0)}")
    print("-" * 80)
    print(f"累计收益率: {metrics.get('cumulative_return', 0)*100:.2f}%")
    print(f"年化收益率: {metrics.get('annualized_return', 0)*100:.2f}%")
    print(f"最大回撤:   {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"组合夏普率: {metrics.get('sharpe_ratio', 0):.2f}")
    print("=" * 80)


def print_period_comparison(nav_series, btc_price_series, target_symbol='BTC_USDT'):
    """
    打印分时段 NAV vs 标的收益对比及超额收益
    - 每个自然年
    - 最近15个自然月
    """
    symbol_label = target_symbol.replace('_', ' ')
    dates = nav_series.index

    def period_return(series):
        if len(series) < 2 or series.iloc[0] == 0:
            return float('nan')
        return series.iloc[-1] / series.iloc[0] - 1

    print("\n" + "=" * 80)
    print("📅 分年度收益对比".center(80))
    print("=" * 80)
    print(f"{'年份':<8} {'NAV收益':>10} {f'{symbol_label}收益':>10} {'超额收益':>10}")
    print("-" * 80)

    years = sorted(set(d.year for d in dates))
    for year in years:
        mask = dates.year == year
        nav_year = nav_series[mask]
        price_year = btc_price_series[mask]

        nav_ret = period_return(nav_year)
        price_ret = period_return(price_year)
        excess = nav_ret - price_ret if not (pd.isna(nav_ret) or pd.isna(price_ret)) else float('nan')

        nav_str = f"{nav_ret*100:+.2f}%" if not pd.isna(nav_ret) else "N/A"
        price_str = f"{price_ret*100:+.2f}%" if not pd.isna(price_ret) else "N/A"
        exc_str = f"{excess*100:+.2f}%" if not pd.isna(excess) else "N/A"

        print(f"{year:<8} {nav_str:>10} {price_str:>10} {exc_str:>10}")

    print("=" * 80)

    print("\n" + "=" * 80)
    print("📅 最近15个月收益对比".center(80))
    print("=" * 80)
    print(f"{'月份':<10} {'NAV收益':>10} {f'{symbol_label}收益':>10} {'超额收益':>10}")
    print("-" * 80)

    last_date = dates[-1]
    month_starts = []
    for m_offset in range(14, -1, -1):
        ref = last_date - pd.DateOffset(months=m_offset)
        month_start = pd.Timestamp(year=ref.year, month=ref.month, day=1)
        month_starts.append(month_start)

    for i in range(len(month_starts)):
        m_start = month_starts[i]
        m_end = month_starts[i + 1] if i + 1 < len(month_starts) else (last_date + pd.Timedelta(days=1))

        mask = (dates >= m_start) & (dates < m_end)
        nav_month = nav_series[mask]
        price_month = btc_price_series[mask]

        if len(nav_month) < 1:
            continue

        nav_ret = period_return(nav_month)
        price_ret = period_return(price_month)
        excess = nav_ret - price_ret if not (pd.isna(nav_ret) or pd.isna(price_ret)) else float('nan')

        label = m_start.strftime('%Y-%m')
        nav_str = f"{nav_ret*100:+.2f}%" if not pd.isna(nav_ret) else "N/A"
        price_str = f"{price_ret*100:+.2f}%" if not pd.isna(price_ret) else "N/A"
        exc_str = f"{excess*100:+.2f}%" if not pd.isna(excess) else "N/A"

        print(f"{label:<10} {nav_str:>10} {price_str:>10} {exc_str:>10}")

    print("=" * 80)


def run_portfolio_optimization(portfolio_id: int, mode: str, target_symbol: str = None) -> dict:
    """
    组合回测核心逻辑（从 CLI 交互中解耦，可供 AI Agent 程序化调用）

    Args:
        portfolio_id (int): 组合ID
        mode (str): 权重模式，'equal' / 'sharpe' / 'scaling' / 'risk_parity'
        target_symbol (str): 交易标的，如 'BTC_USDT', 'SPY', 'GCUSD' 等

    Returns:
        dict: 业绩指标字典，失败或数据为空时返回 {}
    """
    try:
        if not target_symbol:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('SELECT target_symbol FROM portfolios WHERE portfolio_id = ?', (portfolio_id,))
            row = cursor.fetchone()
            conn.close()
            target_symbol = row[0] if row and row[0] else 'BTC_USDT'
        
        target_asset = SYMBOL_ASSET_MAP.get(target_symbol, 'crypto')
        logger.info(f"组合 {portfolio_id} 标的: {target_symbol} ({target_asset})")

        strategies = get_portfolio_strategies_with_code(portfolio_id)
        if not strategies:
            logger.error(f"组合 {portfolio_id} 没有找到策略")
            return {}

        logger.info(f"组合 {portfolio_id} 包含 {len(strategies)} 个策略")

        df_daily = load_daily_data_directly(symbol=target_symbol, target_asset=target_asset)
        if df_daily.empty:
            logger.error(f"{target_symbol} 日线数据为空")
            return {}

        df_daily = df_daily.copy()
        df_daily['timestamp'] = df_daily.index
        logger.info(f"日线数据准备完成，共 {len(df_daily)} 个交易日")

        signals_dict = {}
        for strategy in strategies:
            dir_id = strategy['dir_id']
            try:
                signals = generate_strategy_signals(
                    strategy['best_code'],
                    strategy['params'],
                    df_daily
                )
                if signals is not None:
                    signals_dict[dir_id] = signals
            except Exception as e:
                logger.error(f"处理策略 {dir_id} 失败: {e}")

        if not signals_dict:
            logger.error("没有策略成功生成信号")
            return {}

        logger.info(f"共 {len(signals_dict)} 个策略成功生成信号")

        strategy_sharpe_rates = get_strategy_sharpe_rates(strategies)

        price_return = df_daily['close'].pct_change().fillna(0)

        signals_df = pd.DataFrame(signals_dict)
        combined_signal = combine_signals(signals_df, mode, strategy_sharpe_rates, price_return)

        actual_position = combined_signal.shift(1).fillna(0)

        turnover = actual_position.diff().fillna(0).abs()
        fee_paid = turnover * 0.0005

        daily_return = actual_position * price_return - fee_paid

        nav = (1 + daily_return).cumprod()
        nav.iloc[0] = 1.0

        result_df = pd.DataFrame({
            'date': df_daily['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'),
            'btc_price': df_daily['close'],
            'combined_signal': combined_signal,
            'turnover': turnover,
            'fee_paid': fee_paid,
            'daily_return': daily_return,
            'nav': nav
        }).reset_index(drop=True)

        inserted_count = save_portfolio_records(portfolio_id, result_df, run_phase='BACKTEST')
        if inserted_count == 0:
            logger.error("保存对账单失败")
            return {}

        metrics = calculate_performance_metrics(nav, daily_return)
        if not metrics:
            logger.error("计算业绩指标失败")
            return {}

        update_portfolio_metrics(portfolio_id, mode, metrics)
        update_portfolio_status(portfolio_id, 'TESTED')

        print_period_comparison(nav, df_daily['close'], target_symbol=target_symbol)

        logger.info(f"组合 {portfolio_id} 回测完成，业绩已持久化")
        return metrics

    except Exception as e:
        logger.error(f"组合回测执行失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    print("=" * 100)
    print("🚀 组合回测计算器 - 极简版".center(100))
    print("=" * 100)
    
    portfolio = select_portfolio()
    if portfolio is None:
        return
    
    portfolio_id = portfolio['portfolio_id']
    portfolio_name = portfolio['name']
    target_symbol = portfolio.get('target_symbol', 'BTC_USDT')
    
    print(f"\n✅ 已选择组合: {portfolio_name} (ID: {portfolio_id}, 标的: {target_symbol})")
    
    strategies = get_portfolio_strategies_with_code(portfolio_id)
    
    if not strategies:
        print("\n❌ 组合没有找到策略！")
        return
    
    print(f"\n✅ 组合包含 {len(strategies)} 个策略")
    
    mode_name, mode = select_weight_mode()
    print(f"\n✅ 已选择: {mode_name}")
    
    metrics = run_portfolio_optimization(portfolio_id, mode, target_symbol=target_symbol)
    
    if not metrics:
        print("\n❌ 组合回测失败！请检查日志")
        return
    
    print_performance_report(len(strategies), metrics, mode_name)
    print(f"\n✅ 组合测试数据已存入数据库")
    print(f"✅ 组合状态已变更为 TESTED")
    print(f"✅ 业绩指标已持久化到数据库")
    print("\n🎉 组合回测完成，可随时接入实盘！")


if __name__ == "__main__":
    main()

