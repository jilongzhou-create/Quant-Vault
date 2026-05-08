#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
OOS Backtest Runner - V6/V7 黄金宏观估值模型 OOS 周期回测

OOS Period: 2020-01-01 ~ 2026-04-30
重点验证:
  - 2022: 暴力加息年 (Fed 紧缩周期)
  - 2023-2024: 黄金脱锚年 (金价与实际利率背离)
"""

import os
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from isolated_macro.core.gold_valuation import GoldMacroTrendV6, GoldMacroTrendV7
from isolated_macro.engine.macro_backtest import MacroBacktestEngine

OOS_START = '2020-01-01'
OOS_END = '2026-04-30'


def run_backtest(df, exposure_col, cost_rate=0.0002, risk_free_rate=0.02):
    bt_df = df[['market_price', exposure_col]].copy()
    bt_df.rename(columns={exposure_col: 'target_exposure'}, inplace=True)
    bt_df = bt_df.dropna()

    engine = MacroBacktestEngine(cost_rate=cost_rate, risk_free_rate=risk_free_rate)
    result, df_detail = engine.run(bt_df)
    return result, df_detail


def print_annual_breakdown(df_detail, label):
    print(f"\n  [{label}] Annual breakdown:")
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


def print_regime_detail(df, year, label):
    yr = df[df.index.year == year]
    if yr.empty:
        print(f"  {year}: No data")
        return

    price_start = yr['market_price'].iloc[0]
    price_end = yr['market_price'].iloc[-1]
    yr_return = (price_end / price_start - 1)

    avg_v6 = yr['target_exposure_v6'].mean()
    avg_v7 = yr['target_exposure_v7'].mean()
    avg_trend = yr['trend_flag'].mean()
    avg_macro_bull = (yr['macro_flag'] == 1).mean()
    avg_macro_neutral = (yr['macro_flag'] == 0).mean()
    avg_macro_bear = (yr['macro_flag'] == -1).mean()

    veto_days = int(((yr['macro_flag'] == -1) & (yr['trend_flag'] == 1)).sum())
    total_days = len(yr)

    print(f"\n  [{label}] {year} Regime Detail:")
    print(f"    Gold Return:      {yr_return:+.2%}")
    print(f"    V6 Avg Exposure:  {avg_v6:.4f}")
    print(f"    V7 Avg Exposure:  {avg_v7:.4f}")
    print(f"    Trend ON:         {avg_trend:.1%} of days")
    print(f"    Macro Bull:       {avg_macro_bull:.1%}  Neutral: {avg_macro_neutral:.1%}  Bear: {avg_macro_bear:.1%}")
    print(f"    Macro Veto Days:  {veto_days}/{total_days} ({veto_days/total_days:.1%})")


def main():
    print("=" * 78)
    print("  Gold Macro Valuation - OOS Backtest (V6 vs V7)")
    print(f"  Period: {OOS_START} ~ {OOS_END}")
    print("=" * 78)

    print("\n[Step 1] Initializing V6 model...")
    model_v6 = GoldMacroTrendV6()
    df_v6 = model_v6.calculate_target_exposure(start_date=OOS_START, end_date=OOS_END)

    print("\n[Step 2] Initializing V7 model...")
    model_v7 = GoldMacroTrendV7()
    df_v7 = model_v7.calculate_target_exposure(start_date=OOS_START, end_date=OOS_END)

    if df_v6.empty or df_v7.empty:
        print("[ERROR] No OOS data! Check database coverage.")
        return

    df = df_v6[['market_price', 'target_exposure_v6']].copy()
    df = df.join(df_v7[['target_exposure_v7']], how='inner')

    print(f"\n[Data] OOS rows: {len(df)}, {df.index[0].date()} ~ {df.index[-1].date()}")

    df['trend_flag'] = df_v6['trend_flag']
    df['macro_flag'] = df_v6['macro_flag']

    print("\n[Step 3] Running OOS backtests (rf=2%)...")
    v6_result, v6_detail = run_backtest(df, 'target_exposure_v6')
    v7_result, v7_detail = run_backtest(df, 'target_exposure_v7')

    MacroBacktestEngine.print_report(v6_result, title="V6 Resonance (OOS 2020-2026, rf=2%)")
    MacroBacktestEngine.print_report(v7_result, title="V7 MacroVeto (OOS 2020-2026, rf=2%)")

    print("\n" + "=" * 78)
    print("  V6 vs V7 OOS Comparison (2020-2026)")
    print("=" * 78)
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
    print(f"  {'Metric':<22s}  {'V6 Resonance':>14s}  {'V7 MacroVeto':>14s}")
    print("-" * 56)
    for label, key, fmt in metrics:
        v6_val = v6_result[key]
        v7_val = v7_result[key]
        if fmt == '%':
            print(f"  {label:<22s}  {v6_val:>13.2%}  {v7_val:>13.2%}")
        else:
            print(f"  {label:<22s}  {v6_val:>14.4f}  {v7_val:>14.4f}")
    print("-" * 56)
    print(f"  {'Benchmark (B&H)':<22s}  {v6_result['market_total_return']:>13.2%}")
    print(f"  {'Benchmark Annual':<22s}  {v6_result['market_annualized']:>13.2%}")
    print("=" * 78)

    print_annual_breakdown(v6_detail, "V6 Resonance")
    print_annual_breakdown(v7_detail, "V7 MacroVeto")

    print("\n[Step 4] Key Regime Analysis (V6 vs V7 Exposure):")
    df['year'] = df.index.year
    for year in sorted(df['year'].unique()):
        yr_data = df[df['year'] == year]
        if yr_data.empty:
            continue
        avg_v6 = yr_data['target_exposure_v6'].mean()
        avg_v7 = yr_data['target_exposure_v7'].mean()
        price_start = yr_data['market_price'].iloc[0]
        price_end = yr_data['market_price'].iloc[-1]
        yr_return = (price_end / price_start - 1)
        print(f"  {year}: Gold {yr_return:+.1%}  V6={avg_v6:.3f}  V7={avg_v7:.3f}")

    print_regime_detail(df, 2022, "Rate Hike Year")
    print_regime_detail(df, 2023, "Gold Decoupling")
    print_regime_detail(df, 2024, "Gold Decoupling II")

    print(f"\n[Step 5] OOS Summary:")
    print(f"  V6 Strategy:  Total={v6_result['total_return']:>10.2%}  "
          f"Annual={v6_result['annualized_return']:>10.2%}  "
          f"Sharpe={v6_result['sharpe_ratio']:.4f}  MaxDD={v6_result['max_drawdown']:.2%}")
    print(f"  V7 Strategy:  Total={v7_result['total_return']:>10.2%}  "
          f"Annual={v7_result['annualized_return']:>10.2%}  "
          f"Sharpe={v7_result['sharpe_ratio']:.4f}  MaxDD={v7_result['max_drawdown']:.2%}")
    print(f"  Buy&Hold:     Total={v6_result['market_total_return']:>10.2%}  "
          f"Annual={v6_result['market_annualized']:>10.2%}")

    print("\n" + "=" * 78)
    print("  OOS Backtest Complete!")
    print("=" * 78)


if __name__ == '__main__':
    main()
