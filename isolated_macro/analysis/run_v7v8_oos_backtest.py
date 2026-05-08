#!/usr/bin/env python3
"""
OOS Backtest Runner - V7/V8 黄金宏观估值模型 OOS 周期回测

OOS Period: 2020-01-01 ~ 2026-04-30
重点验证:
  - 2022: 暴力加息年 (V7 宏观否决防守完美)
  - 2023-2024: 黄金脱锚年 (V8 微观/脱锚分能否修复踏空)
"""

import os
import sys
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from isolated_macro.core.gold_valuation import GoldMacroTrendV7, GoldMacroTrendV8
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


def print_regime_detail(df_v7, df_v8, year, label):
    yr7 = df_v7[df_v7.index.year == year]
    yr8 = df_v8[df_v8.index.year == year]
    if yr7.empty or yr8.empty:
        print(f"  {year}: No data")
        return

    price_start = yr7['market_price'].iloc[0]
    price_end = yr7['market_price'].iloc[-1]
    yr_return = (price_end / price_start - 1)

    avg_v7 = yr7['target_exposure_v7'].mean()
    avg_v8 = yr8['target_exposure_v8'].mean()
    avg_trend = yr7['trend_flag'].mean()

    print(f"\n  [{label}] {year} Regime Detail:")
    print(f"    Gold Return:      {yr_return:+.2%}")
    print(f"    V7 Avg Exposure:  {avg_v7:.4f}")
    print(f"    V8 Avg Exposure:  {avg_v8:.4f}")
    print(f"    Trend ON:         {avg_trend:.1%} of days")

    if 'macro_flag' in yr7.columns:
        avg_macro_bull = (yr7['macro_flag'] == 1).mean()
        avg_macro_bear = (yr7['macro_flag'] == -1).mean()
        veto_days = int(((yr7['macro_flag'] == -1) & (yr7['trend_flag'] == 1)).sum())
        print(f"    V7 Macro Bull:    {avg_macro_bull:.1%}  Bear: {avg_macro_bear:.1%}  Veto: {veto_days}d")

    if 'S_fin' in yr8.columns:
        s_fin_pos = (yr8['S_fin'] > 0).mean()
        s_fin_neg = (yr8['S_fin'] < 0).mean()
        s_liq_pos = (yr8['S_liq'] > 0).mean()
        s_mic_pos = (yr8['S_mic'] > 0).mean()
        avg_score = yr8['net_macro_score'].mean()
        veto_days = int(((yr8['net_macro_score'] < 0) & (yr8['trend_flag'] == 1)).sum())
        print(f"    V8 S_fin+:       {s_fin_pos:.1%}  S_fin-: {s_fin_neg:.1%}")
        print(f"    V8 S_liq+:       {s_liq_pos:.1%}  S_mic+: {s_mic_pos:.1%}")
        print(f"    V8 Avg Score:    {avg_score:+.3f}  Multi-Veto: {veto_days}d")


def main():
    log_path = os.path.join(os.path.dirname(__file__), 'v7v8_oos_log.txt')
    log_file = open(log_path, 'w', encoding='utf-8')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log("=" * 78)
    log("  Gold Macro Valuation - OOS Backtest (V7 vs V8)")
    log(f"  Period: {OOS_START} ~ {OOS_END}")
    log("=" * 78)

    log("\n[Step 1] Initializing V7 model...")
    model_v7 = GoldMacroTrendV7()
    df_v7 = model_v7.calculate_target_exposure(start_date=OOS_START, end_date=OOS_END)

    log("\n[Step 2] Initializing V8 model...")
    model_v8 = GoldMacroTrendV8()
    df_v8 = model_v8.calculate_target_exposure(start_date=OOS_START, end_date=OOS_END)

    if df_v7.empty or df_v8.empty:
        log("[ERROR] No OOS data! Check database coverage.")
        log_file.close()
        return

    df = df_v7[['market_price', 'target_exposure_v7', 'trend_flag', 'macro_flag']].copy()
    df = df.join(df_v8[['target_exposure_v8', 'S_fin', 'S_liq', 'S_mic', 'net_macro_score']], how='inner')

    log(f"\n[Data] OOS rows: {len(df)}, {df.index[0].date()} ~ {df.index[-1].date()}")

    log("\n[Step 3] Running OOS backtests (rf=2%)...")
    v7_result, v7_detail = run_backtest(df, 'target_exposure_v7')
    v8_result, v8_detail = run_backtest(df, 'target_exposure_v8')

    bh_df = df[['market_price']].copy()
    bh_df['target_exposure'] = 1.0
    bh_engine = MacroBacktestEngine(cost_rate=0.0002, risk_free_rate=0.02)
    bh_result, _ = bh_engine.run(bh_df.dropna())

    log("\n" + "=" * 78)
    log("  V7 vs V8 OOS Comparison (2020-2026)")
    log("=" * 78)
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
    log(f"  {'Metric':<22s}  {'V7 MacroVeto':>14s}  {'V8 PillarScr':>14s}  {'Buy&Hold':>14s}")
    log("-" * 70)
    for label, key, fmt in metrics:
        v7_val = v7_result[key]
        v8_val = v8_result[key]
        bh_val = bh_result[key]
        if fmt == '%':
            log(f"  {label:<22s}  {v7_val:>13.2%}  {v8_val:>13.2%}  {bh_val:>13.2%}")
        else:
            log(f"  {label:<22s}  {v7_val:>14.4f}  {v8_val:>14.4f}  {bh_val:>14.4f}")
    log("=" * 78)

    print_annual_breakdown(v7_detail, "V7 MacroVeto")
    print_annual_breakdown(v8_detail, "V8 PillarScoring")

    log("\n[Step 4] Key Regime Analysis:")
    for year in sorted(df.index.year.unique()):
        yr = df[df.index.year == year]
        if yr.empty:
            continue
        avg_v7 = yr['target_exposure_v7'].mean()
        avg_v8 = yr['target_exposure_v8'].mean()
        price_start = yr['market_price'].iloc[0]
        price_end = yr['market_price'].iloc[-1]
        yr_return = (price_end / price_start - 1)
        log(f"  {year}: Gold {yr_return:+.1%}  V7={avg_v7:.3f}  V8={avg_v8:.3f}")

    print_regime_detail(df_v7, df_v8, 2022, "Rate Hike Year")
    print_regime_detail(df_v7, df_v8, 2023, "Gold Decoupling I")
    print_regime_detail(df_v7, df_v8, 2024, "Gold Decoupling II")

    log(f"\n[Step 5] OOS Summary:")
    log(f"  V7 Strategy:  Total={v7_result['total_return']:>10.2%}  "
        f"Annual={v7_result['annualized_return']:>10.2%}  "
        f"Sharpe={v7_result['sharpe_ratio']:.4f}  MaxDD={v7_result['max_drawdown']:.2%}")
    log(f"  V8 Strategy:  Total={v8_result['total_return']:>10.2%}  "
        f"Annual={v8_result['annualized_return']:>10.2%}  "
        f"Sharpe={v8_result['sharpe_ratio']:.4f}  MaxDD={v8_result['max_drawdown']:.2%}")
    log(f"  Buy&Hold:     Total={bh_result['total_return']:>10.2%}  "
        f"Annual={bh_result['annualized_return']:>10.2%}")

    log("\n" + "=" * 78)
    log("  OOS Backtest Complete!")
    log("=" * 78)

    log_file.close()


if __name__ == '__main__':
    main()
