#!/usr/bin/env python3
"""
Framework IS Backtest - 底座 + 卫星因子 IS 回测 (2007-2019)

测试 CreditPanicFactor 和 SgePremiumFactor 的 IC,
若通过则由 DynamicSynthesizer 自动合成权重。
重点输出 2008, 2009, 2013, 2018, 2019 五个关键年份。
"""

import os
import sys
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from isolated_macro.core.gold_valuation import GoldMacroTrendV7
from isolated_macro.core.gold_macro_framework import GoldMacroFramework
from isolated_macro.engine.macro_backtest import MacroBacktestEngine

IS_START = '2007-01-01'
IS_END = '2019-12-31'
KEY_YEARS = [2008, 2009, 2013, 2018, 2019]


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


def print_ic_report(df, label):
    from isolated_macro.core.framework.synthesizer import FactorResearcher

    print(f"\n  [{label}] Satellite IC Report (v3: Forward Cumulative Residual):")
    print(f"  {'Factor':<16s}  {'Trigger%':>9s}  {'GlobalIC':>10s}  {'CondIC':>10s}  {'HitRate':>9s}  {'AvgWt':>9s}  {'Verdict':>10s}")
    print("  " + "-" * 80)

    gold_return = df['market_price'].pct_change()
    core_expected = df['core_signal'].shift(1) * gold_return
    residual = (gold_return - core_expected.fillna(0)).fillna(0)

    for sat_name in ['credit_panic', 'sge_premium']:
        sig_col = f'{sat_name}_signal'
        ic_col = f'{sat_name}_ic'
        wt_col = f'{sat_name}_weight'
        if sig_col not in df.columns:
            continue
        sig = df[sig_col]
        ic = df[ic_col]
        wt = df[wt_col]
        trigger_pct = sig.mean()
        avg_ic = ic.mean()
        avg_wt = wt.mean()

        eval_result = FactorResearcher.evaluate_factor(sig, residual, name=sat_name)
        cond_ic = eval_result['conditional_ic']
        hit_rate = eval_result['hit_rate']
        accepted = eval_result['accepted']
        fp = eval_result['forward_period']

        verdict = "ACCEPTED" if accepted else "REJECTED"
        print(f"  {sat_name:<16s}  {trigger_pct:>8.2%}  {avg_ic:>+10.4f}  {cond_ic:>+10.4f}  {hit_rate:>8.1%}  {avg_wt:>9.4f}  [{verdict}]")
        print(f"  {'(forward=' + str(fp) + 'd)':<16s}  {'':>9s}  {'':>10s}  {'':>10s}  {'':>9s}  {'':>9s}")

    df['year'] = df.index.year
    print(f"\n  Yearly IC Breakdown:")
    print(f"  {'Year':>6s}  {'credit_panic_ic':>16s}  {'sge_premium_ic':>16s}")
    print("  " + "-" * 44)
    for year, grp in df.groupby('year'):
        cp_ic = grp['credit_panic_ic'].mean() if 'credit_panic_ic' in grp.columns else 0
        sp_ic = grp['sge_premium_ic'].mean() if 'sge_premium_ic' in grp.columns else 0
        print(f"  {year:>6d}  {cp_ic:>+16.4f}  {sp_ic:>+16.4f}")


def main():
    print("=" * 78)
    print("  Core + Satellites IS Backtest (2007-2019)")
    print("  Mode: SMA Voting + CreditPanic + SgePremium + DynamicSynthesizer")
    print(f"  Period: {IS_START} ~ {IS_END}")
    print("=" * 78)

    print("\n[Step 1] Running V7 (Macro Veto) - IS baseline...")
    model_v7 = GoldMacroTrendV7()
    df_v7 = model_v7.calculate_target_exposure(start_date=IS_START, end_date=IS_END)

    print("\n[Step 2] Running Core + Satellites Framework - IS...")
    framework = GoldMacroFramework()
    df_fw = framework.run(start_date=IS_START, end_date=IS_END)

    if df_fw.empty:
        print("[ERROR] Framework produced no data!")
        return

    print_ic_report(df_fw, "IS Full Period")

    print("\n[Step 3] Running IS backtests (rf=2%)...")
    v7_result, v7_detail = run_backtest(df_v7, 'target_exposure_v7')
    fw_result, fw_detail = run_backtest(df_fw, 'target_exposure')

    bh_df = df_fw[['market_price']].copy()
    bh_df['target_exposure'] = 1.0
    bh_engine = MacroBacktestEngine(cost_rate=0.0002, risk_free_rate=0.02)
    bh_result, _ = bh_engine.run(bh_df.dropna())

    print("\n" + "=" * 78)
    print("  V7 vs Core+Satellites IS Comparison (2007-2019)")
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
        ('Total Turnover', 'total_turnover', ''),
    ]
    print(f"  {'Metric':<22s}  {'V7 Veto':>12s}  {'Core+Sat':>12s}  {'Buy&Hold':>12s}")
    print("-" * 64)
    for label, key, fmt in metrics:
        v7_val = v7_result[key]
        fw_val = fw_result[key]
        bh_val = bh_result[key]
        if fmt == '%':
            print(f"  {label:<22s}  {v7_val:>11.2%}  {fw_val:>11.2%}  {bh_val:>11.2%}")
        else:
            print(f"  {label:<22s}  {v7_val:>12.4f}  {fw_val:>12.4f}  {bh_val:>12.4f}")
    print("=" * 78)

    print_annual_breakdown(v7_detail, "V7 MacroVeto")
    print_annual_breakdown(fw_detail, "Core+Satellites")

    print("\n[Step 4] Vote + Satellite Distribution Analysis:")
    df = df_fw.copy()
    for year in sorted(df.index.year.unique()):
        yr = df[df.index.year == year]
        if yr.empty:
            continue
        price_start = yr['market_price'].iloc[0]
        price_end = yr['market_price'].iloc[-1]
        yr_return = (price_end / price_start - 1)
        avg_core = yr['core_signal'].mean()
        avg_total = yr['total_score'].mean()
        avg_exp = yr['target_exposure'].mean()
        trend_pct = yr['trend_intact'].mean()
        regime_vc = yr['score_regime'].value_counts()
        top_regime = regime_vc.index[0] if len(regime_vc) > 0 else 'N/A'
        cp_trig = yr['credit_panic_signal'].mean() if 'credit_panic_signal' in yr.columns else 0
        sp_trig = yr['sge_premium_signal'].mean() if 'sge_premium_signal' in yr.columns else 0
        print(f"  {year}: Gold {yr_return:+.1%}  core={avg_core:+.3f}  "
              f"total={avg_total:+.3f}  exp={avg_exp:.3f}  "
              f"trend={trend_pct:.1%}  CP={cp_trig:.1%}  SP={sp_trig:.1%}  "
              f"regime={top_regime}")

    print("\n[Step 5] Key Year Deep Dive:")
    for year in KEY_YEARS:
        yr7 = df_v7[df_v7.index.year == year]
        yrfw = df_fw[df_fw.index.year == year]
        if yr7.empty or yrfw.empty:
            continue
        price_start = yr7['market_price'].iloc[0]
        price_end = yr7['market_price'].iloc[-1]
        yr_return = (price_end / price_start - 1)
        avg_v7 = yr7['target_exposure_v7'].mean()
        avg_fw = yrfw['target_exposure'].mean()
        avg_core = yrfw['core_signal'].mean()
        avg_total = yrfw['total_score'].mean()
        trend_pct = yrfw['trend_intact'].mean()
        cp_trig = yrfw['credit_panic_signal'].mean() if 'credit_panic_signal' in yrfw.columns else 0
        sp_trig = yrfw['sge_premium_signal'].mean() if 'sge_premium_signal' in yrfw.columns else 0
        cp_wt = yrfw['credit_panic_weight'].mean() if 'credit_panic_weight' in yrfw.columns else 0
        sp_wt = yrfw['sge_premium_weight'].mean() if 'sge_premium_weight' in yrfw.columns else 0
        print(f"  {year}: Gold {yr_return:+.1%}")
        print(f"         V7_exp={avg_v7:.3f}  FW_exp={avg_fw:.3f}  "
              f"core={avg_core:+.3f}  total={avg_total:+.3f}  trend={trend_pct:.1%}")
        print(f"         CP: trigger={cp_trig:.1%}  weight={cp_wt:.4f}")
        print(f"         SP: trigger={sp_trig:.1%}  weight={sp_wt:.4f}")

    print("\n" + "=" * 78)
    print("  Core + Satellites IS Backtest Complete!")
    print("=" * 78)


if __name__ == '__main__':
    main()
