#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
IS Backtest Runner - 黄金宏观估值模型 IS 周期回测 (Smart Trend Edition)

四版对比:
  V1  (纯均值回归):  FinalExposure = -clip(Z / z_cap, -1, 1)
  V2  (缩减敞口):    FinalExposure = RawExposure × ScalingScore
  V2b (趋势融合):    FinalExposure = RawExposure × Scale + Trend × (1-Scale)
  V3  (趋势主导):    FinalExposure = (1-w) × RawExposure + w × TrendSignal
                     w = clip(|trend| / threshold, 0, 1)
"""

import os
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from isolated_macro.core.gold_valuation import GoldValuationModel, GoldMacroTrendV6, GoldMacroTrendV7
from isolated_macro.engine.macro_backtest import MacroBacktestEngine

IS_START = '2007-01-01'
IS_END = '2019-12-31'

VERSIONS = {
    'V1 Mean-Rev': 'target_exposure_v1',
    'V2 Scale': 'target_exposure_v2',
    'V2b Fusion': 'target_exposure_v2b',
    'V3 TrendDom': 'target_exposure_v3',
    'V4 LongOnly': 'target_exposure_v4',
    'V5 PureTrend': 'target_exposure_v5',
    'V6 Resonance': 'target_exposure_v6',
    'V7 MacroVeto': 'target_exposure_v7',
}


def run_backtest(df, exposure_col, cost_rate=0.0002, risk_free_rate=0.0):
    """通用回测函数"""
    bt_df = df[['market_price', exposure_col]].copy()
    bt_df.rename(columns={exposure_col: 'target_exposure'}, inplace=True)
    bt_df = bt_df.dropna()

    engine = MacroBacktestEngine(cost_rate=cost_rate, risk_free_rate=risk_free_rate)
    result, df_detail = engine.run(bt_df)
    return result, df_detail


def print_comparison(results_dict):
    """打印多版对比表"""
    versions = list(results_dict.keys())
    n = len(versions)
    col_w = 14

    print("\n" + "=" * (26 + (col_w + 2) * n))
    print("  Multi-Version IS Comparison (2007-2019)")
    print("=" * (26 + (col_w + 2) * n))

    header = f"  {'Metric':<22s}"
    for v in versions:
        header += f"  {v:>{col_w}s}"
    print(header)
    print("-" * (26 + (col_w + 2) * n))

    metrics = [
        ('Total Return', 'total_return', '%'),
        ('Annual Return', 'annualized_return', '%'),
        ('Annual Vol', 'annualized_vol', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Sharpe (rf=0)', 'sharpe_ratio_rf', ''),
        ('Max Drawdown', 'max_drawdown', '%'),
        ('Calmar Ratio', 'calmar_ratio', ''),
        ('Win Rate', 'win_rate', '%'),
        ('Profit/Loss', 'profit_loss_ratio', ''),
        ('Total Turnover', 'total_turnover', ''),
    ]

    for label, key, fmt in metrics:
        row = f"  {label:<22s}"
        for v in versions:
            val = results_dict[v][key]
            if fmt == '%':
                row += f"  {val:>{col_w - 1}.2%}"
            else:
                row += f"  {val:>{col_w}.4f}"
        print(row)

    print("-" * (26 + (col_w + 2) * n))
    bh_ret = results_dict[versions[0]]['market_total_return']
    bh_ann = results_dict[versions[0]]['market_annualized']
    print(f"  {'Benchmark (B&H)':<22s}  {bh_ret:>{col_w - 1}.2%}")
    print(f"  {'Benchmark Annual':<22s}  {bh_ann:>{col_w - 1}.2%}")
    print("=" * (26 + (col_w + 2) * n))


def print_annual_breakdown(df_detail, label):
    """打印年度分解"""
    print(f"\n  [{label}] Annual breakdown:")
    print(f"  {'Year':>6s}  {'Return':>10s}  {'Sharpe':>8s}  {'MaxDD':>10s}  {'Exposure':>10s}")
    print("  " + "-" * 52)

    df_detail['year'] = df_detail.index.year
    for year, grp in df_detail.groupby('year'):
        yr_ret = (1 + grp['strategy_return']).prod() - 1
        yr_vol = grp['strategy_return'].std() * np.sqrt(252)
        yr_sharpe = (yr_ret / yr_vol) if yr_vol > 0 else 0

        cum_max = grp['cum_strategy_return'].cummax()
        yr_dd = ((grp['cum_strategy_return'] - cum_max) / cum_max).min()

        avg_exp = grp['position'].mean()

        print(f"  {year:>6d}  {yr_ret:>10.2%}  {yr_sharpe:>8.2f}  {yr_dd:>10.2%}  {avg_exp:>10.4f}")


def main():
    print("=" * 78)
    print("  Gold Macro Valuation - IS Backtest (8-Version Comparison)")
    print(f"  Period: {IS_START} ~ {IS_END}")
    print("=" * 78)

    # Step 1: 初始化估值模型 (V1~V5)
    print("\n[Step 1] Initializing GoldValuationModel...")
    model = GoldValuationModel(model_id='gold_macro_v1')

    # Step 2: 计算目标敞口 (含 V1~V5)
    print(f"\n[Step 2] Computing target exposure for IS period...")
    df = model.calculate_target_exposure(start_date=IS_START, end_date=IS_END)

    if df.empty:
        print("[ERROR] No valuation data! Aborting.")
        return

    # Step 2b: 初始化 V6 模型并计算敞口
    print(f"\n[Step 2b] Initializing GoldMacroTrendV6...")
    model_v6 = GoldMacroTrendV6()
    df_v6 = model_v6.calculate_target_exposure(start_date=IS_START, end_date=IS_END)

    if not df_v6.empty:
        df = df.join(df_v6[['target_exposure_v6']], how='left')
        df['target_exposure_v6'] = df['target_exposure_v6'].fillna(0.0)
        print(f"  V6 exposure merged. Rows: {len(df)}")
    else:
        print("[WARN] V6 produced no data, filling with 0")
        df['target_exposure_v6'] = 0.0

    # Step 2c: 初始化 V7 模型并计算敞口
    print(f"\n[Step 2c] Initializing GoldMacroTrendV7...")
    model_v7 = GoldMacroTrendV7()
    df_v7 = model_v7.calculate_target_exposure(start_date=IS_START, end_date=IS_END)

    if not df_v7.empty:
        df = df.join(df_v7[['target_exposure_v7']], how='left')
        df['target_exposure_v7'] = df['target_exposure_v7'].fillna(0.0)
        print(f"  V7 exposure merged. Rows: {len(df)}")
    else:
        print("[WARN] V7 produced no data, filling with 0")
        df['target_exposure_v7'] = 0.0

    # Step 3: 打印详细序列
    model.print_zscore_exposure_series(df, n_tail=15)

    # Step 4: 运行八版回测 (V6/V7 使用 rf=2%)
    print("\n[Step 3] Running backtests for all versions...")
    print("  V1~V5: risk_free_rate=0% | V6~V7: risk_free_rate=2%")
    all_results = {}
    all_details = {}

    for ver_name, col_name in VERSIONS.items():
        rf = 0.02 if ver_name in ('V6 Resonance', 'V7 MacroVeto') else 0.0
        print(f"\n  {ver_name} (rf={rf:.0%})...")
        result, detail = run_backtest(df, col_name, risk_free_rate=rf)
        all_results[ver_name] = result
        all_details[ver_name] = detail
        MacroBacktestEngine.print_report(result, title=f"{ver_name} (IS 2007-2019, rf={rf:.0%})")

    # Step 5: 对比表
    print_comparison(all_results)

    # Step 6: V3 年度分解
    print_annual_breakdown(all_details['V3 TrendDom'], "V3 Trend-Dominant")

    # Step 6b: V6 年度分解
    print_annual_breakdown(all_details['V6 Resonance'], "V6 Macro-Trend Resonance")

    # Step 6c: V7 年度分解
    print_annual_breakdown(all_details['V7 MacroVeto'], "V7 Macro Veto")

    # Step 7: 关键场景分析
    print("\n[Step 4] Key Regime Analysis:")
    df['year'] = df.index.year

    for year in [2008, 2011, 2013, 2014, 2015, 2016, 2019]:
        yr_data = df[df['year'] == year]
        if yr_data.empty:
            continue
        avg_v3 = yr_data['target_exposure_v3'].mean()
        avg_v6 = yr_data['target_exposure_v6'].mean()
        avg_v7 = yr_data['target_exposure_v7'].mean()
        price_start = yr_data['market_price'].iloc[0]
        price_end = yr_data['market_price'].iloc[-1]
        yr_return = (price_end / price_start - 1)

        print(f"  {year}: Gold {yr_return:+.1%}  "
              f"V3={avg_v3:+.3f}  V6={avg_v6:.3f}  V7={avg_v7:.3f}")

    # Step 8: V3/V6/V7 与 B&H 对比
    v3r = all_results['V3 TrendDom']
    v6r = all_results['V6 Resonance']
    v7r = all_results['V7 MacroVeto']
    print(f"\n[Step 5] V3 / V6 / V7 vs Buy-and-Hold:")
    print(f"  V3 Strategy:  Total={v3r['total_return']:>10.2%}  "
          f"Annual={v3r['annualized_return']:>10.2%}  Sharpe={v3r['sharpe_ratio']:.4f}  MaxDD={v3r['max_drawdown']:.2%}")
    print(f"  V6 Strategy:  Total={v6r['total_return']:>10.2%}  "
          f"Annual={v6r['annualized_return']:>10.2%}  Sharpe={v6r['sharpe_ratio']:.4f}  MaxDD={v6r['max_drawdown']:.2%}")
    print(f"  V7 Strategy:  Total={v7r['total_return']:>10.2%}  "
          f"Annual={v7r['annualized_return']:>10.2%}  Sharpe={v7r['sharpe_ratio']:.4f}  MaxDD={v7r['max_drawdown']:.2%}")
    print(f"  Buy&Hold:     Total={v3r['market_total_return']:>10.2%}  "
          f"Annual={v3r['market_annualized']:>10.2%}")

    print("\n" + "=" * 78)
    print("  IS Backtest Complete!")
    print("=" * 78)


if __name__ == '__main__':
    main()
