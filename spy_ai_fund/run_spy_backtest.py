#!/usr/bin/env python3
"""
策略回测脚本 - SPY AI Fund IS/OOS 回测

用法:
  python spy_ai_fund/run_spy_backtest.py              # IS + OOS 全量回测
  python spy_ai_fund/run_spy_backtest.py --period is   # 仅 IS
  python spy_ai_fund/run_spy_backtest.py --period oos  # 仅 OOS
  python spy_ai_fund/run_spy_backtest.py --period full # IS + OOS 连续回测
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import numpy as np
import pandas as pd

from spy_ai_fund.config import IS_START, IS_END, OOS_START, OOS_END
from spy_ai_fund.core.spy_macro_framework import SpyMacroFramework
from spy_ai_fund.engine.spy_backtest import SpyBacktestEngine


def init_spy_tables():
    """初始化 SPY 数据库表"""
    import sqlite3
    from config import DB_PATH

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_data_spy (
        symbol          TEXT        NOT NULL,
        timestamp       DATETIME    NOT NULL,
        date            TEXT        NOT NULL,
        open            REAL,
        high            REAL,
        low             REAL,
        close           REAL,
        adj_close       REAL        NOT NULL,
        volume          REAL,
        rsi_14          REAL,
        macd            REAL,
        macd_signal     REAL,
        macd_hist       REAL,
        PRIMARY KEY (symbol, timestamp)
    )
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_spy_timestamp ON market_data_spy (timestamp DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_spy_date ON market_data_spy (date)')

    conn.commit()
    conn.close()
    print("[DB] SPY market_data_spy 表初始化完成")


def load_data(start_date, end_date):
    fw = SpyMacroFramework()
    df = fw.run(start_date=start_date, end_date=end_date)
    return df


def run_backtest(df, exposure_col='final_exposure', cost_rate=0.0002, risk_free_rate=0.0):
    if exposure_col not in df.columns:
        exposure_col = 'target_exposure'
    bt_df = df[['market_price', exposure_col]].copy()
    bt_df.rename(columns={exposure_col: 'target_exposure'}, inplace=True)
    bt_df = bt_df.dropna()
    engine = SpyBacktestEngine(cost_rate=cost_rate, risk_free_rate=risk_free_rate)

    rf_series = None
    if 'dtb3' in df.columns:
        rf_series = df['dtb3'].reindex(bt_df.index)

    result, df_detail = engine.run(bt_df, rf_series=rf_series)
    return result, df_detail


def print_annual_breakdown(df_detail, label):
    print(f"\n  [{label}] Annual Breakdown:")
    print(f"  {'Year':>6s}  {'Return':>10s}  {'Sharpe':>8s}  {'MaxDD':>10s}  {'Exposure':>10s}")
    print("  " + "-" * 52)
    df_detail['year'] = df_detail.index.year
    for year, grp in df_detail.groupby('year'):
        yr_ret = (1 + grp['strategy_return']).prod() - 1
        yr_vol = grp['strategy_return'].std() * np.sqrt(252)
        yr_sharpe = (yr_ret / yr_vol) if yr_vol > 1e-10 else 0
        cum_max = grp['cum_strategy_return'].cummax()
        yr_dd = ((grp['cum_strategy_return'] - cum_max) / cum_max).min()
        avg_exp = grp['position'].mean()
        print(f"  {year:>6d}  {yr_ret:>10.2%}  {yr_sharpe:>8.2f}  {yr_dd:>10.2%}  {avg_exp:>10.4f}")


def print_comparison(core_result, bh_result, title):
    metrics = [
        ('Total Return', 'total_return', '%'),
        ('Annual Return', 'annualized_return', '%'),
        ('Annual Vol', 'annualized_vol', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Sharpe (rf=0)', 'sharpe_ratio_rf', ''),
        ('Max Drawdown', 'max_drawdown', '%'),
        ('Calmar Ratio', 'calmar_ratio', ''),
        ('Win Rate', 'win_rate', '%'),
        ('Total Turnover', 'total_turnover', ''),
    ]
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")
    print(f"  {'Metric':<22s}  {'Strategy':>12s}  {'Buy&Hold':>12s}")
    print("-" * 52)
    for label, key, fmt in metrics:
        strat_val = core_result[key]
        bh_val = bh_result[key]
        if fmt == '%':
            print(f"  {label:<22s}  {strat_val:>11.2%}  {bh_val:>11.2%}")
        else:
            print(f"  {label:<22s}  {strat_val:>12.4f}  {bh_val:>12.4f}")
    print(f"{'='*78}")


def run_period_backtest(period_name, start_date, end_date):
    print(f"\n{'#'*78}")
    print(f"  {period_name} Backtest: {start_date} ~ {end_date}")
    print(f"{'#'*78}")

    df = load_data(start_date, end_date)
    if df.empty:
        print("  [ERROR] No data loaded!")
        return None

    core_result, core_detail = run_backtest(df, 'final_exposure')

    bh_df = df[['market_price']].copy()
    bh_df['target_exposure'] = 1.0
    bh_engine = SpyBacktestEngine(cost_rate=0.0002, risk_free_rate=0.0)

    bh_rf_series = None
    if 'dtb3' in df.columns:
        bh_rf_series = df['dtb3'].reindex(bh_df.index)

    bh_result, _ = bh_engine.run(bh_df.dropna(), rf_series=bh_rf_series)

    print_comparison(core_result, bh_result,
                     f"Strategy vs Buy&Hold {period_name} ({start_date}~{end_date})")

    print_annual_breakdown(core_detail, "Strategy")

    return {
        'core_result': core_result,
        'bh_result': bh_result,
        'core_detail': core_detail,
        'df': df,
    }


def main():
    parser = argparse.ArgumentParser(description='SPY AI Fund Strategy Backtest')
    parser.add_argument('--period', choices=['is', 'oos', 'full', 'both'], default='both',
                        help='回测期间: is=IS(2007-2019), oos=OOS(2020-2026), full=全量, both=IS+OOS分别 (默认: both)')
    parser.add_argument('--output', type=str, default=None,
                        help='将回测结果写入指定文件')
    args = parser.parse_args()

    _output_file = None
    if args.output:
        _output_file = open(args.output, 'w', encoding='utf-8')
        _orig_print = __builtins__.print if hasattr(__builtins__, 'print') else print

        def _tee_print(*a, **kw):
            _orig_print(*a, **kw)
            kw2 = dict(kw)
            kw2.pop('file', None)
            kw2['file'] = _output_file
            _orig_print(*a, **kw2)

        import builtins
        builtins.print = _tee_print

    init_spy_tables()

    if args.period == 'is':
        run_period_backtest("IS", IS_START, IS_END)
    elif args.period == 'oos':
        run_period_backtest("OOS", OOS_START, OOS_END)
    elif args.period == 'full':
        run_period_backtest("Full Range", IS_START, OOS_END)
    elif args.period == 'both':
        run_period_backtest("IS", IS_START, IS_END)
        run_period_backtest("OOS", OOS_START, OOS_END)

    print(f"\n{'='*78}")
    print("  SPY Backtest Complete!")
    print(f"{'='*78}")

    if _output_file:
        import builtins
        builtins.print = _orig_print
        _output_file.close()
        print(f"  Results written to {args.output}")


if __name__ == '__main__':
    main()
