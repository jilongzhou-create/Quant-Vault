#!/usr/bin/env python3
"""
IS Factor Verdict + IS/OOS Independent Backtest

核心原则:
  - IS 和 OOS 分别独立运行框架, 避免全区间运行导致信号泄露
  - 所有绩效指标由 MacroBacktestEngine 统一计算, 不做手动拆分
  - Sharpe = ann_ret / ann_vol (引擎标准, 不减 rf)
  - Calmar = ann_ret / |max_dd|
"""

import os, sys
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from isolated_macro.core.gold_valuation import GoldMacroTrendV7
from isolated_macro.core.gold_macro_framework import GoldMacroFramework
from isolated_macro.core.framework.synthesizer import FactorResearcher
from isolated_macro.engine.macro_backtest import MacroBacktestEngine

IS_START = '2007-01-01'
IS_END = '2019-12-31'
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
    print(f"  {'Year':>6s}  {'Return':>10s}  {'Vol':>10s}  {'Sharpe':>8s}  {'MaxDD':>10s}  {'Exposure':>10s}")
    print("  " + "-" * 62)
    df_detail = df_detail.copy()
    df_detail['year'] = df_detail.index.year
    for year, grp in df_detail.groupby('year'):
        yr_ret = (1 + grp['strategy_return']).prod() - 1
        yr_vol = grp['strategy_return'].std() * np.sqrt(252)
        yr_sharpe = yr_ret / yr_vol if yr_vol > 1e-10 else 0
        cum = grp['cum_strategy_return']
        cum_max = cum.cummax()
        yr_dd = ((cum - cum_max) / cum_max).min()
        avg_exp = grp['position'].mean()
        print(f"  {year:>6d}  {yr_ret:>10.2%}  {yr_vol:>10.2%}  {yr_sharpe:>8.2f}  {yr_dd:>10.2%}  {avg_exp:>10.4f}")


def print_metrics_table(results_dict, title):
    print(f"\n  {title}")
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
    names = list(results_dict.keys())
    header = f"  {'Metric':<22s}  " + "  ".join(f"{n:>12s}" for n in names)
    print(header)
    print("  " + "-" * (24 + 14 * len(names)))
    for label, key, fmt in metrics:
        vals = [results_dict[n][key] for n in names]
        if fmt == '%':
            line = f"  {label:<22s}  " + "  ".join(f"{v:>11.2%}" for v in vals)
        else:
            line = f"  {label:<22s}  " + "  ".join(f"{v:>12.4f}" for v in vals)
        print(line)


def main():
    # ================================================================
    # Step 1: IS Factor Verdict
    # ================================================================
    print("=" * 78)
    print("  Step 1: IS (2007-2019) Factor Verdict")
    print("  Evaluation: Conditional IC + Hit Rate + Trigger Rate")
    print("=" * 78)

    framework_is = GoldMacroFramework()
    df_is = framework_is.run(start_date=IS_START, end_date=IS_END)

    if df_is.empty:
        print("[ERROR] No IS data!")
        return

    gold_return = df_is['market_price'].pct_change()
    core_expected = df_is['core_signal'].shift(1) * gold_return
    residual = (gold_return - core_expected.fillna(0)).fillna(0)

    print(f"\n  {'Factor':<16s}  {'Trigger%':>9s}  {'GlobalIC':>10s}  {'CondIC':>10s}  {'HitRate':>9s}  {'Verdict':>10s}")
    print("  " + "-" * 70)

    all_accepted = True
    for name in ['credit_panic', 'sge_premium']:
        sig_col = f'{name}_signal'
        if sig_col not in df_is.columns:
            print(f"  {name:<16s}  SIGNAL NOT FOUND")
            all_accepted = False
            continue
        sig = df_is[sig_col]
        trigger_pct = sig.mean()
        r = FactorResearcher.evaluate_factor(sig, residual, name=name)
        verdict = "ACCEPTED" if r['accepted'] else "DEAD"
        if not r['accepted']:
            all_accepted = False
        print(f"  {name:<16s}  {trigger_pct:>8.2%}  {r['global_ic']:>+10.4f}  {r['conditional_ic']:>+10.4f}  {r['hit_rate']:>8.1%}  [{verdict}]")

    print(f"\n  IS Verdict Summary: {'ALL ACCEPTED' if all_accepted else 'HAS DEAD FACTOR'}")

    if not all_accepted:
        print("\n  [STOP] 有因子未通过 IS 评估, 不进入 OOS 回测。")
        return

    # ================================================================
    # Step 2: IS Backtest (独立运行)
    # ================================================================
    print("\n" + "=" * 78)
    print("  Step 2: IS Backtest (2007-2019) - Independent Run")
    print("=" * 78)

    model_v7_is = GoldMacroTrendV7()
    df_v7_is = model_v7_is.calculate_target_exposure(start_date=IS_START, end_date=IS_END)

    v7_is_result, v7_is_detail = run_backtest(df_v7_is, 'target_exposure_v7')
    fw_is_result, fw_is_detail = run_backtest(df_is, 'target_exposure')

    bh_is_df = df_is[['market_price']].copy()
    bh_is_df['target_exposure'] = 1.0
    bh_is_engine = MacroBacktestEngine(cost_rate=0.0002, risk_free_rate=0.02)
    bh_is_result, _ = bh_is_engine.run(bh_is_df.dropna())

    print_metrics_table(
        {'V7 Veto': v7_is_result, 'Core+Sat': fw_is_result, 'Buy&Hold': bh_is_result},
        "IS (2007-2019) Performance"
    )

    print_annual_breakdown(v7_is_detail, "V7 MacroVeto (IS)")
    print_annual_breakdown(fw_is_detail, "Core+Satellites (IS)")

    # ================================================================
    # Step 3: OOS Backtest (独立运行, 信号不泄露)
    # ================================================================
    print("\n" + "=" * 78)
    print("  Step 3: OOS Backtest (2020-2026) - Independent Run")
    print("  NOTE: Framework runs independently on OOS data only")
    print("=" * 78)

    model_v7_oos = GoldMacroTrendV7()
    df_v7_oos = model_v7_oos.calculate_target_exposure(start_date=OOS_START, end_date=OOS_END)

    framework_oos = GoldMacroFramework()
    df_oos = framework_oos.run(start_date=OOS_START, end_date=OOS_END)

    if df_oos.empty:
        print("[ERROR] No OOS data!")
        return

    v7_oos_result, v7_oos_detail = run_backtest(df_v7_oos, 'target_exposure_v7')
    fw_oos_result, fw_oos_detail = run_backtest(df_oos, 'target_exposure')

    bh_oos_df = df_oos[['market_price']].copy()
    bh_oos_df['target_exposure'] = 1.0
    bh_oos_engine = MacroBacktestEngine(cost_rate=0.0002, risk_free_rate=0.02)
    bh_oos_result, _ = bh_oos_engine.run(bh_oos_df.dropna())

    print_metrics_table(
        {'V7 Veto': v7_oos_result, 'Core+Sat': fw_oos_result, 'Buy&Hold': bh_oos_result},
        "OOS (2020-2026) Performance"
    )

    print_annual_breakdown(v7_oos_detail, "V7 MacroVeto (OOS)")
    print_annual_breakdown(fw_oos_detail, "Core+Satellites (OOS)")

    # ================================================================
    # Step 4: OOS Factor IC Report
    # ================================================================
    print("\n" + "=" * 78)
    print("  Step 4: OOS Factor IC Report (Information Only)")
    print("=" * 78)

    gold_ret_oos = df_oos['market_price'].pct_change()
    core_exp_oos = df_oos['core_signal'].shift(1) * gold_ret_oos
    res_oos = (gold_ret_oos - core_exp_oos.fillna(0)).fillna(0)

    print(f"\n  {'Factor':<16s}  {'Trigger%':>9s}  {'GlobalIC':>10s}  {'CondIC':>10s}  {'HitRate':>9s}  {'AvgWt':>9s}  {'Verdict':>10s}")
    print("  " + "-" * 80)
    for sat_name in ['credit_panic', 'sge_premium']:
        sig_col = f'{sat_name}_signal'
        ic_col = f'{sat_name}_ic'
        wt_col = f'{sat_name}_weight'
        if sig_col not in df_oos.columns:
            continue
        sig = df_oos[sig_col]
        ic = df_oos[ic_col]
        wt = df_oos[wt_col]
        trigger_pct = sig.mean()
        avg_ic = ic.mean()
        avg_wt = wt.mean()
        r = FactorResearcher.evaluate_factor(sig, res_oos, name=sat_name)
        verdict = "ACCEPTED" if r['accepted'] else "DEAD"
        print(f"  {sat_name:<16s}  {trigger_pct:>8.2%}  {avg_ic:>+10.4f}  {r['conditional_ic']:>+10.4f}  {r['hit_rate']:>8.1%}  {avg_wt:>9.4f}  [{verdict}]")

    # ================================================================
    # Step 5: Summary
    # ================================================================
    print("\n" + "=" * 78)
    print("  Step 5: IS vs OOS Summary (Engine Metrics)")
    print("=" * 78)

    print(f"\n  {'Period':<14s}  {'Strategy':<12s}  {'AnnRet':>10s}  {'AnnVol':>10s}  {'Sharpe':>8s}  {'Calmar':>8s}  {'MaxDD':>10s}")
    print("  " + "-" * 80)

    for period_label, v7_r, fw_r in [
        ('IS 2007-2019', v7_is_result, fw_is_result),
        ('OOS 2020-2026', v7_oos_result, fw_oos_result),
    ]:
        for strat_label, r in [('V7 Veto', v7_r), ('Core+Sat', fw_r)]:
            print(f"  {period_label:<14s}  {strat_label:<12s}  {r['annualized_return']:>10.2%}  {r['annualized_vol']:>10.2%}  {r['sharpe_ratio']:>8.4f}  {r['calmar_ratio']:>8.4f}  {r['max_drawdown']:>10.2%}")

    print("\n" + "=" * 78)
    print("  Backtest Complete!")
    print("=" * 78)


if __name__ == '__main__':
    main()
