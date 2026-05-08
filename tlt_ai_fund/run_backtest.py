#!/usr/bin/env python3
"""
策略回测脚本 - TLT AI Fund IS/OOS 回测 (V13 Core + Satellite)

用法:
  python tlt_ai_fund/run_backtest.py              # IS + OOS 全量回测
  python tlt_ai_fund/run_backtest.py --period is   # 仅 IS
  python tlt_ai_fund/run_backtest.py --period oos  # 仅 OOS
  python tlt_ai_fund/run_backtest.py --period full # IS + OOS 连续回测
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import numpy as np
import pandas as pd

from tlt_ai_fund.config import (
    IS_START, IS_END, OOS_START, OOS_END,
    ZSCORE_WINDOW, ZSCORE_THRESHOLD, SAT_ASYM_CAP, CORE_BULL_PROTECT,
)
from tlt_ai_fund.core.tlt_macro_framework import TltMacroFramework
from tlt_ai_fund.engine.tlt_backtest import TltBacktestEngine, synthesize_final_exposure
from tlt_ai_fund.db.schema import get_accepted_factors, init_ai_fund_tables
from tlt_ai_fund.agents.factor_miner import load_factor_instance
from tlt_ai_fund.agents.gatekeeper import Gatekeeper


def init_tlt_tables():
    from config import DB_PATH
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_data_tlt (
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

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tlt_timestamp ON market_data_tlt (timestamp DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tlt_date ON market_data_tlt (date)')

    conn.commit()
    conn.close()
    print("[DB] TLT market_data_tlt table ready")


def load_data(start_date, end_date):
    fw = TltMacroFramework()
    df = fw.run(start_date=start_date, end_date=end_date)
    df = Gatekeeper._enrich_with_data_lake(df, start_date, end_date)
    return df


def load_ai_factor_signals(df, accepted_factors):
    signals = {}
    for af in accepted_factors:
        try:
            instance = load_factor_instance(af['factor_id'], af['source_file'])
            sig = instance.calculate_signal(df)
            signals[af['factor_id']] = sig
        except Exception as e:
            print(f"  [WARN] Failed to load factor {af['factor_id']}: {e}")
    return signals


def _extract_direction(factor_id: str) -> str:
    for d in ['unstructured', 'microstructure', 'volatility']:
        if d in factor_id:
            return d
    return 'unknown'


def compute_composite_signal(df, core_signal, factor_signals):
    if not factor_signals:
        return core_signal.copy()

    signal_df = pd.DataFrame(factor_signals).reindex(df.index).fillna(0.0)
    factor_directions = {fid: _extract_direction(fid) for fid in factor_signals}

    z_window = ZSCORE_WINDOW
    z_threshold = ZSCORE_THRESHOLD
    sat_cap = SAT_ASYM_CAP
    core_protect = CORE_BULL_PROTECT

    zscore_df = pd.DataFrame(index=df.index, columns=list(factor_signals.keys()))
    for fid in factor_signals.keys():
        sig = signal_df[fid]
        rolling_mean = sig.rolling(z_window, min_periods=60).mean()
        rolling_std = sig.rolling(z_window, min_periods=60).std()
        zscore_df[fid] = (sig - rolling_mean) / (rolling_std + 1e-9)

    zscore_df = zscore_df.fillna(0.0)

    awakened_mask = zscore_df.abs() >= z_threshold

    unique_directions = set(factor_directions.values())
    direction_factor_map = {}
    for d in unique_directions:
        direction_factor_map[d] = [fid for fid in factor_signals if factor_directions[fid] == d]

    selected_mask = pd.DataFrame(False, index=df.index, columns=list(factor_signals.keys()))
    for d, fids in direction_factor_map.items():
        if len(fids) == 1:
            selected_mask[fids[0]] = awakened_mask[fids[0]]
        else:
            sub_z = zscore_df[fids].abs()
            max_z_val = sub_z.max(axis=1)
            for fid in fids:
                selected_mask[fid] = awakened_mask[fid] & (sub_z[fid] == max_z_val)

    active_signal = signal_df * selected_mask.astype(float)
    sat_total = active_signal.sum(axis=1)
    sat_total = sat_total.clip(-sat_cap, sat_cap)

    bull_mask = core_signal >= core_protect
    sat_total = sat_total.where(~(bull_mask & (sat_total < 0)), 0.0)

    total_score = (core_signal + sat_total).clip(0.0, 1.0)

    n_active = selected_mask.sum(axis=1)
    trigger_rate = (n_active > 0).mean()
    print(f"  [Z-Score Pulse] threshold={z_threshold}, window={z_window}: "
          f"avg active factors={n_active.mean():.2f}, "
          f"trigger rate={trigger_rate:.2%}, "
          f"sat mean={sat_total.mean():.4f}, "
          f"sat std={sat_total.std():.4f}")

    return total_score


def run_backtest(df, exposure_col='final_exposure', cost_rate=0.0002, risk_free_rate=0.0):
    if exposure_col not in df.columns:
        exposure_col = 'target_exposure'
    bt_df = df[['market_price', exposure_col]].copy()
    bt_df.rename(columns={exposure_col: 'target_exposure'}, inplace=True)
    bt_df = bt_df.dropna()
    engine = TltBacktestEngine(cost_rate=cost_rate, risk_free_rate=risk_free_rate)

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


def print_comparison(core_result, fw_result, bh_result, title):
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
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"  {'Metric':<22s}  {'Pure Core':>12s}  {'Core+Sat':>12s}  {'Buy&Hold':>12s}")
    print("-" * 64)
    for label, key, fmt in metrics:
        core_val = core_result[key]
        fw_val = fw_result[key]
        bh_val = bh_result[key]
        if fmt == '%':
            print(f"  {label:<22s}  {core_val:>11.2%}  {fw_val:>11.2%}  {bh_val:>11.2%}")
        else:
            print(f"  {label:<22s}  {core_val:>12.4f}  {fw_val:>12.4f}  {bh_val:>12.4f}")
    print(f"{'='*90}")


def print_factor_summary(accepted_factors):
    if not accepted_factors:
        print("  (no accepted satellite factors)")
        return
    print(f"\n  Production Satellite Factors:")
    print(f"  {'Factor ID':<45s}  {'Direction':<18s}  {'Method':<15s}")
    print("  " + "-" * 80)
    for af in accepted_factors:
        print(f"  {af['factor_id']:<45s}  {af['mining_direction']:<18s}  {af['mining_method']:<15s}")


def run_period_backtest(period_name, start_date, end_date, accepted_factors):
    print(f"\n{'#'*78}")
    print(f"  {period_name} Backtest: {start_date} ~ {end_date}")
    print(f"  AI Satellite Factors: {len(accepted_factors)}")
    print(f"{'#'*78}")

    df = load_data(start_date, end_date)
    if df.empty:
        print("  [ERROR] No data loaded!")
        return None

    factor_signals = load_ai_factor_signals(df, accepted_factors)

    core_col = 'target_exposure' if 'target_exposure' in df.columns else 'tlt_core_signal'
    if core_col not in df.columns:
        core_col = 'core_signal'
    core_signal = df[core_col]

    print(f"  Computing composite signal (base: {core_col})...")
    total_score = compute_composite_signal(df, core_signal, factor_signals)

    df_core = df.copy()
    df_core['target_exposure'] = df['target_exposure']

    df_fw = df.copy()
    df_fw['target_exposure'] = total_score

    core_result, core_detail = run_backtest(df_core, 'target_exposure')
    fw_result, fw_detail = run_backtest(df_fw, 'target_exposure')

    bh_df = df[['market_price']].copy()
    bh_df['target_exposure'] = 1.0
    bh_engine = TltBacktestEngine(cost_rate=0.0002, risk_free_rate=0.0)

    bh_rf_series = None
    if 'dtb3' in df.columns:
        bh_rf_series = df['dtb3'].reindex(bh_df.index)

    bh_result, _ = bh_engine.run(bh_df.dropna(), rf_series=bh_rf_series)

    print_comparison(core_result, fw_result, bh_result,
                     f"Pure Core vs Core+Sat {period_name} ({start_date}~{end_date})")

    print_annual_breakdown(core_detail, "Pure Core")
    print_annual_breakdown(fw_detail, "Core+Sat")

    return {
        'core_result': core_result,
        'fw_result': fw_result,
        'bh_result': bh_result,
        'core_detail': core_detail,
        'fw_detail': fw_detail,
        'df_fw': df_fw,
        'factor_signals': factor_signals,
    }


def main():
    parser = argparse.ArgumentParser(description='TLT AI Fund Strategy Backtest')
    parser.add_argument('--period', choices=['is', 'oos', 'full', 'both'], default='both',
                        help='Period: is=IS(2007-2019), oos=OOS(2020-2026), full=all, both=IS+OOS (default: both)')
    parser.add_argument('--output', type=str, default=None,
                        help='Write results to file')
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
            _orig_print(*a, kw2)

        import builtins
        builtins.print = _tee_print

    init_tlt_tables()
    init_ai_fund_tables()

    accepted_factors = get_accepted_factors()
    print(f"\n  Production Factor Pool: {len(accepted_factors)} accepted factors")
    if not accepted_factors:
        print("  [WARN] No accepted factors found! Backtest will use Core Anchor only.")
    else:
        print_factor_summary(accepted_factors)

    if args.period == 'is':
        run_period_backtest("IS", IS_START, IS_END, accepted_factors)
    elif args.period == 'oos':
        run_period_backtest("OOS", OOS_START, OOS_END, accepted_factors)
    elif args.period == 'full':
        run_period_backtest("Full Range", IS_START, OOS_END, accepted_factors)
    elif args.period == 'both':
        run_period_backtest("IS", IS_START, IS_END, accepted_factors)
        run_period_backtest("OOS", OOS_START, OOS_END, accepted_factors)

    print(f"\n{'='*78}")
    print("  TLT Backtest Complete!")
    print(f"{'='*78}")

    if _output_file:
        import builtins
        builtins.print = _orig_print
        _output_file.close()
        print(f"  Results written to {args.output}")


if __name__ == '__main__':
    main()
