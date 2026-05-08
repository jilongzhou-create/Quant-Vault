#!/usr/bin/env python3
"""
Portfolio 44 完整回测脚本 - 2018~2026年4月
组合策略: Macro_Momentum_Reversion_V1 (8个BTC策略, scaling权重)
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime
from trading_engine.backtest_engine import load_daily_data_directly, compile_strategy
from database.db_manager import DB_PATH

PORTFOLIO_ID = 44
START_DATE = '2018-01-01'
END_DATE = '2026-04-30'
COST_RATE = 0.0005


def load_portfolio_strategies():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT pc.dir_id, sv.code_content, sv.params_json, sd.name
        FROM portfolio_components pc
        JOIN strategy_directions sd ON pc.dir_id = sd.dir_id
        JOIN strategy_versions sv ON sd.best_version_id = sv.ver_id
        WHERE pc.portfolio_id = ?
          AND sv.run_status != 'OVERFITTED'
    ''', (PORTFOLIO_ID,))
    rows = cursor.fetchall()
    conn.close()

    strategies = []
    for dir_id, code, params_json, name in rows:
        try:
            params = json.loads(params_json) if params_json else {}
        except:
            params = {}
        strategies.append({
            'dir_id': dir_id,
            'code': code,
            'params': params,
            'name': name,
        })
    return strategies


def generate_signals_for_strategy(strategy, df):
    try:
        func = compile_strategy(strategy['code'])
        signals = func(df.copy(), strategy['params'])
        if isinstance(signals, pd.Series):
            signals = signals.reindex(df.index).fillna(0)
            return signals
        else:
            print(f"  [WARN] {strategy['name']} did not return Series")
            return pd.Series(0, index=df.index)
    except Exception as e:
        print(f"  [ERROR] {strategy['name']}: {e}")
        return pd.Series(0, index=df.index)


def combine_signals_scaling(signals_df):
    raw_sum = signals_df.sum(axis=1)
    max_positive = raw_sum[raw_sum > 0].max() if (raw_sum > 0).any() else 1.0
    max_negative = raw_sum[raw_sum < 0].min() if (raw_sum < 0).any() else -1.0

    def scale_signal(val):
        if val > 0:
            return val / max_positive
        elif val < 0:
            return val / abs(max_negative)
        else:
            return 0.0

    combined = raw_sum.apply(scale_signal).clip(-1, 1)
    return combined


def run_backtest(df, combined_signal, start_date=None, end_date=None):
    mask = pd.Series(True, index=df.index)
    if start_date:
        mask &= (df.index >= start_date)
    if end_date:
        mask &= (df.index <= end_date)

    df_sub = df[mask].copy()
    signal_sub = combined_signal[mask].copy()

    if len(df_sub) < 10:
        return None

    price_return = df_sub['close'].pct_change().fillna(0)
    actual_position = signal_sub.shift(1).fillna(0)
    turnover = actual_position.diff().fillna(0).abs()
    fee_paid = turnover * COST_RATE
    daily_return = actual_position * price_return - fee_paid
    nav = (1 + daily_return).cumprod()
    nav.iloc[0] = 1.0

    return {
        'df': df_sub,
        'signal': signal_sub,
        'daily_return': daily_return,
        'nav': nav,
        'price_return': price_return,
        'actual_position': actual_position,
    }


def calc_metrics(result, label=""):
    if result is None:
        return None
    nav = result['nav']
    dr = result['daily_return']
    n_days = len(nav)
    if n_days < 2:
        return None

    total_return = nav.iloc[-1] - 1
    years = n_days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    peak = nav.expanding(min_periods=1).max()
    dd = (nav - peak) / peak
    max_dd = dd.min()

    daily_std = dr.std()
    sharpe = (dr.mean() / daily_std) * np.sqrt(365.25) if daily_std > 1e-9 else 0
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-9 else 0

    return {
        'label': label,
        'start': nav.index[0].strftime('%Y-%m-%d'),
        'end': nav.index[-1].strftime('%Y-%m-%d'),
        'days': n_days,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
    }


def calc_yearly_metrics(result):
    if result is None:
        return []
    nav = result['nav']
    dr = result['daily_return']
    years_data = []

    for year, grp in dr.groupby(dr.index.year):
        yr_ret = (1 + grp).prod() - 1
        yr_vol = grp.std() * np.sqrt(365.25)
        yr_sharpe = (grp.mean() / grp.std()) * np.sqrt(365.25) if grp.std() > 1e-9 else 0

        nav_grp = nav[nav.index.year == year]
        peak = nav_grp.expanding(min_periods=1).max()
        yr_dd = ((nav_grp - peak) / peak).min()

        yr_calmar = yr_ret / abs(yr_dd) if abs(yr_dd) > 1e-9 else 0
        avg_pos = result['actual_position'][result['actual_position'].index.year == year].mean()

        years_data.append({
            'year': year,
            'return': yr_ret,
            'vol': yr_vol,
            'sharpe': yr_sharpe,
            'max_dd': yr_dd,
            'calmar': yr_calmar,
            'avg_position': avg_pos,
        })

    return years_data


def main():
    print("=" * 80)
    print("  Portfolio 44 Full Backtest: 2018-01-01 ~ 2026-04-30")
    print("  Strategy: Macro_Momentum_Reversion_V1 (8 strategies, scaling weight)")
    print("=" * 80)

    strategies = load_portfolio_strategies()
    print(f"\n  Loaded {len(strategies)} strategies:")
    for s in strategies:
        print(f"    - {s['name']}")

    print("\n  Loading BTC daily data...")
    df = load_daily_data_directly(symbol='BTC_USDT', target_asset='crypto')
    if df.empty:
        print("  [ERROR] No data loaded!")
        return

    df = df[df.index >= START_DATE]
    df = df[df.index <= END_DATE + ' 23:59:59']
    print(f"  Data range: {df.index[0]} ~ {df.index[-1]}, {len(df)} trading days")

    print("\n  Generating signals for each strategy...")
    signals_dict = {}
    for s in strategies:
        print(f"    Computing: {s['name']}...")
        sig = generate_signals_for_strategy(s, df)
        signals_dict[s['name']] = sig
        nonzero = (sig.abs() > 1e-6).sum()
        print(f"      Non-zero signals: {nonzero}/{len(sig)} ({nonzero/len(sig)*100:.1f}%)")

    signals_df = pd.DataFrame(signals_dict, index=df.index)
    print(f"\n  Combining signals (scaling mode)...")
    combined_signal = combine_signals_scaling(signals_df)

    nonzero_combined = (combined_signal.abs() > 1e-6).sum()
    print(f"  Combined signal non-zero: {nonzero_combined}/{len(combined_signal)} ({nonzero_combined/len(combined_signal)*100:.1f}%)")
    print(f"  Combined signal mean: {combined_signal.mean():.4f}, std: {combined_signal.std():.4f}")
    print(f"  Combined signal range: [{combined_signal.min():.4f}, {combined_signal.max():.4f}]")

    # === Full range backtest ===
    print("\n" + "=" * 80)
    print("  FULL RANGE BACKTEST: 2018-01-01 ~ 2026-04-30")
    print("=" * 80)

    result_full = run_backtest(df, combined_signal)
    metrics_full = calc_metrics(result_full, "Full Range")
    if metrics_full:
        print(f"  Total Return:    {metrics_full['total_return']:>10.2%}")
        print(f"  Annual Return:   {metrics_full['annual_return']:>10.2%}")
        print(f"  Sharpe Ratio:    {metrics_full['sharpe']:>10.4f}")
        print(f"  Max Drawdown:    {metrics_full['max_dd']:>10.2%}")
        print(f"  Calmar Ratio:    {metrics_full['calmar']:>10.4f}")

    # === 2018-2021 April backtest ===
    print("\n" + "=" * 80)
    print("  FOCUS PERIOD: 2018-01-01 ~ 2021-04-30")
    print("=" * 80)

    result_early = run_backtest(df, combined_signal, start_date='2018-01-01', end_date='2021-04-30')
    metrics_early = calc_metrics(result_early, "2018-2021.04")
    if metrics_early:
        print(f"  Total Return:    {metrics_early['total_return']:>10.2%}")
        print(f"  Annual Return:   {metrics_early['annual_return']:>10.2%}")
        print(f"  Sharpe Ratio:    {metrics_early['sharpe']:>10.4f}")
        print(f"  Max Drawdown:    {metrics_early['max_dd']:>10.2%}")
        print(f"  Calmar Ratio:    {metrics_early['calmar']:>10.4f}")

    # === Buy & Hold benchmark ===
    print("\n" + "=" * 80)
    print("  BUY & HOLD BENCHMARK")
    print("=" * 80)

    bh_signal = pd.Series(1.0, index=df.index)
    result_bh_full = run_backtest(df, bh_signal)
    metrics_bh_full = calc_metrics(result_bh_full, "B&H Full")
    if metrics_bh_full:
        print(f"  [Full Range] Total Return: {metrics_bh_full['total_return']:>10.2%}, "
              f"Annual: {metrics_bh_full['annual_return']:>10.2%}, "
              f"Sharpe: {metrics_bh_full['sharpe']:>10.4f}, "
              f"MaxDD: {metrics_bh_full['max_dd']:>10.2%}, "
              f"Calmar: {metrics_bh_full['calmar']:>10.4f}")

    result_bh_early = run_backtest(df, bh_signal, start_date='2018-01-01', end_date='2021-04-30')
    metrics_bh_early = calc_metrics(result_bh_early, "B&H 2018-2021.04")
    if metrics_bh_early:
        print(f"  [2018-2021.04] Total Return: {metrics_bh_early['total_return']:>10.2%}, "
              f"Annual: {metrics_bh_early['annual_return']:>10.2%}, "
              f"Sharpe: {metrics_bh_early['sharpe']:>10.4f}, "
              f"MaxDD: {metrics_bh_early['max_dd']:>10.2%}, "
              f"Calmar: {metrics_bh_early['calmar']:>10.4f}")

    # === Yearly breakdown ===
    print("\n" + "=" * 80)
    print("  YEARLY BREAKDOWN (Portfolio 44)")
    print("=" * 80)
    print(f"  {'Year':>6s}  {'Return':>10s}  {'Vol':>10s}  {'Sharpe':>8s}  {'MaxDD':>10s}  {'Calmar':>8s}  {'AvgPos':>8s}")
    print("  " + "-" * 70)

    yearly = calc_yearly_metrics(result_full)
    for y in yearly:
        print(f"  {y['year']:>6d}  {y['return']:>10.2%}  {y['vol']:>10.2%}  {y['sharpe']:>8.2f}  {y['max_dd']:>10.2%}  {y['calmar']:>8.2f}  {y['avg_position']:>8.4f}")

    # === Yearly breakdown - Buy & Hold ===
    print("\n" + "=" * 80)
    print("  YEARLY BREAKDOWN (Buy & Hold)")
    print("=" * 80)
    print(f"  {'Year':>6s}  {'Return':>10s}  {'Vol':>10s}  {'Sharpe':>8s}  {'MaxDD':>10s}  {'Calmar':>8s}")
    print("  " + "-" * 58)

    yearly_bh = calc_yearly_metrics(result_bh_full)
    for y in yearly_bh:
        print(f"  {y['year']:>6d}  {y['return']:>10.2%}  {y['vol']:>10.2%}  {y['sharpe']:>8.2f}  {y['max_dd']:>10.2%}  {y['calmar']:>8.2f}")

    # === Comparison table ===
    print("\n" + "=" * 80)
    print("  COMPARISON SUMMARY")
    print("=" * 80)
    print(f"  {'Metric':<22s}  {'Portfolio44':>12s}  {'Buy&Hold':>12s}")
    print("  " + "-" * 50)
    if metrics_full and metrics_bh_full:
        for label, m, mbh in [
            ('Total Return', metrics_full['total_return'], metrics_bh_full['total_return']),
            ('Annual Return', metrics_full['annual_return'], metrics_bh_full['annual_return']),
            ('Sharpe Ratio', metrics_full['sharpe'], metrics_bh_full['sharpe']),
            ('Max Drawdown', metrics_full['max_dd'], metrics_bh_full['max_dd']),
            ('Calmar Ratio', metrics_full['calmar'], metrics_bh_full['calmar']),
        ]:
            if label in ('Total Return', 'Annual Return', 'Max Drawdown'):
                print(f"  {label:<22s}  {m:>11.2%}  {mbh:>11.2%}")
            else:
                print(f"  {label:<22s}  {m:>12.4f}  {mbh:>12.4f}")

    print("\n  === 2018-2021.04 Focus Period ===")
    print(f"  {'Metric':<22s}  {'Portfolio44':>12s}  {'Buy&Hold':>12s}")
    print("  " + "-" * 50)
    if metrics_early and metrics_bh_early:
        for label, m, mbh in [
            ('Total Return', metrics_early['total_return'], metrics_bh_early['total_return']),
            ('Annual Return', metrics_early['annual_return'], metrics_bh_early['annual_return']),
            ('Sharpe Ratio', metrics_early['sharpe'], metrics_bh_early['sharpe']),
            ('Max Drawdown', metrics_early['max_dd'], metrics_bh_early['max_dd']),
            ('Calmar Ratio', metrics_early['calmar'], metrics_bh_early['calmar']),
        ]:
            if label in ('Total Return', 'Annual Return', 'Max Drawdown'):
                print(f"  {label:<22s}  {m:>11.2%}  {mbh:>11.2%}")
            else:
                print(f"  {label:<22s}  {m:>12.4f}  {mbh:>12.4f}")

    # === Individual strategy contribution ===
    print("\n" + "=" * 80)
    print("  INDIVIDUAL STRATEGY CONTRIBUTION")
    print("=" * 80)
    print(f"  {'Strategy':<45s}  {'NonZero%':>9s}  {'MeanSig':>8s}  {'FullSharpe':>10s}")
    print("  " + "-" * 80)

    for name, sig in signals_dict.items():
        nonzero_pct = (sig.abs() > 1e-6).mean()
        mean_sig = sig.mean()
        sig_result = run_backtest(df, sig)
        if sig_result:
            sig_dr = sig_result['daily_return']
            sig_sharpe = (sig_dr.mean() / sig_dr.std()) * np.sqrt(365.25) if sig_dr.std() > 1e-9 else 0
        else:
            sig_sharpe = 0
        print(f"  {name:<45s}  {nonzero_pct:>8.1%}  {mean_sig:>+8.4f}  {sig_sharpe:>10.2f}")

    print("\n" + "=" * 80)
    print("  Backtest Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
