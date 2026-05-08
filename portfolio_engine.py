#!/usr/bin/env python3
"""
Portfolio Engine - All-Weather Portfolio Backtesting with IS/OOS Isolation

Usage:
    python portfolio_engine.py --allocator V1DynamicEqualWeight --period is
    python portfolio_engine.py --allocator V1DynamicEqualWeight --period full
    python portfolio_engine.py --allocator V1DynamicEqualWeight --period oos --run-oos

IS period: 2007-01-01 ~ 2019-12-31
OOS period: 2020-01-01 ~ 2026-12-31
"""

import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
from datetime import datetime

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

IS_START = '2007-01-01'
IS_END = '2019-12-31'
OOS_START = '2020-01-01'
OOS_END = '2026-04-30'

IS_SHARPE_GATE = 1.0
IS_CALMAR_GATE = 0.8

COST_RATE = 0.0002

OUTPUT_DIR = os.path.join(project_root, 'all_weather_portfolio', 'output')

ASSET_NAMES = ['GOLD', 'TLT', 'SPY', 'BTC']

CACHE_DIR = os.path.join(project_root, 'all_weather_portfolio', '_strategy_cache')


def _get_db_fingerprint():
    import sqlite3
    db_path = os.path.join(project_root, 'data', 'trading_system_prod.db')
    if not os.path.exists(db_path):
        return 'no_db'
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        tables_to_check = [
            ('raw_data', 'SELECT COUNT(*) FROM raw_data'),
            ('factor_data', 'SELECT COUNT(*) FROM factor_data'),
            ('gold_ai_factor_registry', "SELECT COUNT(*) FROM gold_ai_factor_registry WHERE status='accepted'"),
            ('tlt_ai_factor_registry', "SELECT COUNT(*) FROM tlt_ai_factor_registry WHERE status='accepted'"),
            ('spy_ai_factor_registry', "SELECT COUNT(*) FROM spy_ai_factor_registry WHERE status='accepted'"),
            ('btc_ai_factor_registry', "SELECT COUNT(*) FROM btc_ai_factor_registry WHERE status='accepted'"),
        ]
        parts = []
        for name, query in tables_to_check:
            try:
                cursor.execute(query)
                count = cursor.fetchone()[0]
                parts.append(f"{name}={count}")
            except Exception:
                parts.append(f"{name}=?")
        conn.close()
        return '|'.join(parts)
    except Exception:
        return 'db_error'


def _get_factor_snapshot():
    import sqlite3
    db_path = os.path.join(project_root, 'data', 'trading_system_prod.db')
    if not os.path.exists(db_path):
        return {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        snapshot = {}
        for asset, table in [('GOLD', 'gold_ai_factor_registry'), ('TLT', 'tlt_ai_factor_registry'),
                              ('SPY', 'spy_ai_factor_registry'), ('BTC', 'btc_ai_factor_registry')]:
            cursor.execute(f"SELECT factor_id FROM {table} WHERE status='accepted' ORDER BY factor_id")
            rows = cursor.fetchall()
            snapshot[asset] = [r[0] for r in rows]
        conn.close()
        return snapshot
    except Exception:
        return {}


def _check_cache_staleness():
    os.makedirs(CACHE_DIR, exist_ok=True)
    current_fp = _get_db_fingerprint()
    any_stale = False
    for asset_name in ASSET_NAMES:
        fingerprint_path = os.path.join(CACHE_DIR, f'{asset_name}_fingerprint.txt')
        if not os.path.exists(fingerprint_path):
            continue
        with open(fingerprint_path, 'r') as f:
            cached_fp = f.read().strip()
        if cached_fp != current_fp:
            any_stale = True
            break

    if any_stale:
        print("\n  " + "!" * 60)
        print("  WARNING: Database has changed since cache was created!")
        print("  Factor counts differ between cache and current DB.")
        print("  Current DB fingerprint: " + current_fp)
        print("  To use latest factors, run with --refresh-cache")
        print("  " + "!" * 60 + "\n")


def _load_cached_strategy(asset_name, start_date, end_date):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f'{asset_name}_cache.csv')

    if not os.path.exists(cache_path):
        return None

    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    df = df[mask]
    if df.empty:
        return None

    result = pd.DataFrame(index=df.index)
    exp_col = 'final_exposure' if 'final_exposure' in df.columns else 'target_exposure'
    result['target_exposure'] = df[exp_col]
    result['market_price'] = df['market_price']
    result['market_return'] = df['market_price'].pct_change()
    if 'dtb3' in df.columns:
        result['rf_rate'] = df['dtb3']

    snapshot_path = os.path.join(CACHE_DIR, f'{asset_name}_factors.txt')
    factor_info = ''
    if os.path.exists(snapshot_path):
        with open(snapshot_path, 'r') as f:
            factors = f.read().strip()
        n_factors = len([l for l in factors.split('\n') if l.strip()])
        factor_info = f', {n_factors} satellite factors'
    print(f"    {asset_name}: loaded from cache ({exp_col}, satellite={'YES' if exp_col == 'final_exposure' else 'NO'}{factor_info})", flush=True)
    return result


def _save_strategy_cache(asset_name, df):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f'{asset_name}_cache.csv')
    fingerprint_path = os.path.join(CACHE_DIR, f'{asset_name}_fingerprint.txt')
    snapshot_path = os.path.join(CACHE_DIR, f'{asset_name}_factors.txt')

    cache_cols = ['market_price', 'target_exposure']
    if 'final_exposure' in df.columns:
        cache_cols.append('final_exposure')
    if 'dtb3' in df.columns:
        cache_cols.append('dtb3')
    df[cache_cols].to_csv(cache_path)

    current_fp = _get_db_fingerprint()
    with open(fingerprint_path, 'w') as f:
        f.write(current_fp)

    snapshot = _get_factor_snapshot()
    with open(snapshot_path, 'w') as f:
        factors = snapshot.get(asset_name, [])
        f.write('\n'.join(factors) if factors else '(no satellite factors)')

    n_factors = len(snapshot.get(asset_name, []))
    print(f"    {asset_name}: cached ({len(df)} rows, {n_factors} satellite factors, fp={current_fp[:50]}...)", flush=True)


def load_gold_strategy(start_date, end_date, use_cache=True):
    if use_cache:
        cached = _load_cached_strategy('GOLD', start_date, end_date)
        if cached is not None:
            return cached

    from gold_ai_fund.run_backtest import load_data, load_ai_factor_signals, compute_composite_signal, apply_execution
    df = load_data(start_date, end_date)
    if df.empty:
        return pd.DataFrame()
    from gold_ai_fund.db.schema import get_accepted_factors
    accepted = get_accepted_factors()
    factor_signals = load_ai_factor_signals(df, accepted)
    core_signal = df['core_signal']
    total_score = compute_composite_signal(df, core_signal, factor_signals, method='zscore_pulse')
    df = apply_execution(df, total_score)
    result = pd.DataFrame(index=df.index)
    result['target_exposure'] = df['target_exposure']
    result['market_price'] = df['market_price']
    result['market_return'] = df['market_price'].pct_change()
    if 'dtb3' in df.columns:
        result['rf_rate'] = df['dtb3']
    print(f"    GOLD: {len(accepted)} satellite factors loaded")

    _save_strategy_cache('GOLD', df)
    return result


def load_tlt_strategy(start_date, end_date, use_cache=True):
    if use_cache:
        cached = _load_cached_strategy('TLT', start_date, end_date)
        if cached is not None:
            return cached

    from tlt_ai_fund.run_backtest import load_data
    df = load_data(start_date, end_date)
    if df.empty:
        return pd.DataFrame()
    result = pd.DataFrame(index=df.index)
    exp_col = 'final_exposure' if 'final_exposure' in df.columns else 'target_exposure'
    result['target_exposure'] = df[exp_col]
    result['market_price'] = df['market_price']
    result['market_return'] = df['market_price'].pct_change()
    if 'dtb3' in df.columns:
        result['rf_rate'] = df['dtb3']
    print(f"    TLT: using '{exp_col}' (satellite={'YES' if exp_col == 'final_exposure' else 'NO'})")

    _save_strategy_cache('TLT', df)
    return result


def load_spy_strategy(start_date, end_date, use_cache=True):
    if use_cache:
        cached = _load_cached_strategy('SPY', start_date, end_date)
        if cached is not None:
            return cached

    from spy_ai_fund.run_spy_backtest import load_data
    df = load_data(start_date, end_date)
    if df.empty:
        return pd.DataFrame()
    result = pd.DataFrame(index=df.index)
    exp_col = 'final_exposure' if 'final_exposure' in df.columns else 'target_exposure'
    result['target_exposure'] = df[exp_col]
    result['market_price'] = df['market_price']
    result['market_return'] = df['market_price'].pct_change()
    if 'dtb3' in df.columns:
        result['rf_rate'] = df['dtb3']
    print(f"    SPY: using '{exp_col}' (satellite={'YES' if exp_col == 'final_exposure' else 'NO'})")

    _save_strategy_cache('SPY', df)
    return result


def load_btc_strategy(start_date, end_date, use_cache=True):
    if use_cache:
        cached = _load_cached_strategy('BTC', start_date, end_date)
        if cached is not None:
            return cached

    old_cache = os.path.join(project_root, 'all_weather_portfolio', '_btc_cache.csv')
    if os.path.exists(old_cache):
        os.remove(old_cache)

    print("    BTC: running BtcMacroFramework (first time, ~2-3 min)...", flush=True)
    import time
    t0 = time.time()
    from btc_ai_fund.core.framework.btc_macro_framework import BtcMacroFramework
    fw = BtcMacroFramework()
    df = fw.run(start_date='2018-01-01', end_date=OOS_END)
    t1 = time.time()
    print(f"    BTC: framework run took {t1-t0:.1f}s", flush=True)

    if df.empty:
        print("    BTC: framework returned empty DataFrame!", flush=True)
        return pd.DataFrame()

    _save_strategy_cache('BTC', df)

    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    df = df[mask]
    result = pd.DataFrame(index=df.index)
    exp_col = 'final_exposure' if 'final_exposure' in df.columns else 'target_exposure'
    result['target_exposure'] = df[exp_col]
    result['market_price'] = df['market_price']
    result['market_return'] = df['market_price'].pct_change()
    if 'dtb3' in df.columns:
        result['rf_rate'] = df['dtb3']
    print(f"    BTC: using '{exp_col}' (satellite={'YES' if exp_col == 'final_exposure' else 'NO'})")
    return result


LOADERS = {
    'GOLD': load_gold_strategy,
    'TLT': load_tlt_strategy,
    'SPY': load_spy_strategy,
    'BTC': load_btc_strategy,
}


def load_all_strategies(start_date, end_date, refresh_cache=False):
    print("=" * 70)
    print("  Loading strategy data for each asset...")
    if refresh_cache:
        print("  *** REFRESH CACHE MODE: all caches will be regenerated ***")
    else:
        _check_cache_staleness()
    print("=" * 70)

    asset_data = {}
    for name, loader in LOADERS.items():
        print(f"\n  Loading {name}...")
        try:
            df = loader(start_date, end_date, use_cache=not refresh_cache)
            if df.empty:
                print(f"    {name}: NO DATA")
            else:
                print(f"    {name}: {len(df)} rows, {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
                asset_data[name] = df
        except Exception as e:
            print(f"    {name}: ERROR - {e}")
            import traceback
            traceback.print_exc()

    return asset_data


def align_data(asset_data, start_date, end_date):
    full_range = pd.date_range(start=start_date, end=end_date, freq='B')

    exposures = pd.DataFrame(index=full_range)
    returns = pd.DataFrame(index=full_range)
    prices = pd.DataFrame(index=full_range)
    rf_series = pd.Series(np.nan, index=full_range)

    for name, df in asset_data.items():
        reindexed = df.reindex(full_range)
        exposures[name] = reindexed['target_exposure']
        returns[name] = reindexed['market_return']
        prices[name] = reindexed['market_price']
        if 'rf_rate' in df.columns:
            rf_candidate = df['rf_rate'].reindex(full_range)
            rf_series = rf_series.combine_first(rf_candidate)

    rf_series = rf_series.ffill().fillna(0.0)

    return full_range, exposures, returns, prices, rf_series


def run_backtest(allocator, full_range, exposures, returns, prices, rf_series):
    asset_names = list(exposures.columns)

    active_count = exposures.notna().sum(axis=1)

    print(f"\n  Active asset count over time:")
    for year in range(full_range[0].year, full_range[-1].year + 1):
        yr_mask = full_range.year == year
        avg_active = active_count[yr_mask].mean()
        if not np.isnan(avg_active):
            print(f"    {year}: {avg_active:.1f} assets active")

    print(f"\n  Running backtest with allocator: {allocator.name}...")
    print(f"  Allocator description: {allocator.description}")

    vol_60d = pd.DataFrame(index=full_range)
    for name in asset_names:
        vol_60d[name] = returns[name].rolling(window=60, min_periods=20).std()

    df_mom = prices.shift(21) / prices.shift(252) - 1.0

    is_eom = pd.Series(False, index=full_range)
    for i in range(len(full_range)):
        if i == len(full_range) - 1:
            is_eom.iloc[i] = True
        elif full_range[i].month != full_range[i + 1].month:
            is_eom.iloc[i] = True

    final_weights = pd.DataFrame(index=full_range)
    for name in asset_names:
        final_weights[name] = 0.0

    n_days = len(full_range)
    print(f"  Total trading days: {n_days}")

    for i, date in enumerate(full_range):
        if i % 500 == 0:
            pct = i / n_days * 100
            print(f"    Progress: {i}/{n_days} ({pct:.0f}%)...", flush=True)

        row_exp = {}
        row_md = {}
        for name in asset_names:
            exp_val = exposures.at[date, name] if name in exposures.columns else np.nan
            if pd.isna(exp_val):
                row_exp[name] = None
            else:
                row_exp[name] = float(exp_val)

            md = {}
            if name in returns.columns:
                ret_val = returns.at[date, name]
                md['return'] = float(ret_val) if not pd.isna(ret_val) else None
            if name in prices.columns:
                price_val = prices.at[date, name]
                md['price'] = float(price_val) if not pd.isna(price_val) else None
            if name in vol_60d.columns:
                vol_val = vol_60d.at[date, name]
                md['vol_60d'] = float(vol_val) if not pd.isna(vol_val) else None
            if name in df_mom.columns:
                mom_val = df_mom.at[date, name]
                md['mom'] = float(mom_val) if not pd.isna(mom_val) else None
            row_md[name] = md

        row_md['is_eom'] = bool(is_eom.at[date])

        weights = allocator.allocate(row_exp, row_md)
        for name in asset_names:
            final_weights.at[date, name] = weights.get(name, 0.0)

    print(f"    Progress: {n_days}/{n_days} (100%)...", flush=True)

    position = final_weights.shift(1).ffill().fillna(0.0)
    position_invested = position.sum(axis=1)

    asset_returns = pd.Series(0.0, index=full_range)
    for name in asset_names:
        asset_returns += position[name] * returns[name].fillna(0.0)

    rf_daily = rf_series / 100.0 / 252
    rf_daily = rf_daily.fillna(0.0)

    cash_portion = 1.0 - position_invested
    cash_returns = cash_portion * rf_daily

    position_change = position.diff().abs().sum(axis=1)
    trade_cost = position_change * COST_RATE

    fund_daily_return = asset_returns + cash_returns - trade_cost
    fund_cum_return = (1 + fund_daily_return).cumprod()

    detail = pd.DataFrame(index=full_range)
    detail['fund_daily_return'] = fund_daily_return
    detail['fund_cum_return'] = fund_cum_return
    detail['total_invested'] = position_invested
    detail['cash_portion'] = cash_portion
    detail['asset_returns'] = asset_returns
    detail['cash_returns'] = cash_returns
    detail['trade_cost'] = trade_cost
    detail['rf_daily'] = rf_daily
    detail['active_assets'] = active_count
    detail['cash_interest_income'] = cash_returns

    for name in asset_names:
        detail[f'w_{name}'] = position[name]
        detail[f'exp_{name}'] = exposures[name]
        detail[f'ret_{name}'] = returns[name]
        detail[f'price_{name}'] = prices[name]
        detail[f'pnl_{name}'] = position[name] * returns[name].fillna(0.0)

    spy_data_idx = 'SPY' in asset_names
    if spy_data_idx:
        spy_ret = returns['SPY'].reindex(full_range).fillna(0.0)
        detail['spy_bh_cum_return'] = (1 + spy_ret).cumprod()
    else:
        detail['spy_bh_cum_return'] = np.nan

    return detail


def compute_metrics(daily_return: pd.Series, cum_return: pd.Series, label: str = '') -> dict:
    valid_mask = daily_return.notna() & cum_return.notna()
    daily_valid = daily_return[valid_mask]
    cum_valid = cum_return[valid_mask]

    if len(daily_valid) < 10:
        return {}

    total_return = cum_valid.iloc[-1] - 1
    n_days = len(daily_valid)
    years = n_days / 252
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annual_vol = daily_valid.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 1e-10 else 0

    running_max = cum_valid.cummax()
    drawdown = (cum_valid - running_max) / running_max
    max_dd = drawdown.min()
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0

    return {
        'label': label,
        'total_return': total_return,
        'annualized_return': annual_return,
        'annualized_vol': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
    }


def print_metrics(metrics: dict):
    if not metrics:
        return
    print(f"\n  [{metrics['label']}] Portfolio Performance:")
    print(f"    Total Return:       {metrics['total_return']:>10.2%}")
    print(f"    Annualized Return:  {metrics['annualized_return']:>10.2%}")
    print(f"    Annualized Vol:     {metrics['annualized_vol']:>10.2%}")
    print(f"    Sharpe Ratio:       {metrics['sharpe_ratio']:>10.4f}")
    print(f"    Max Drawdown:       {metrics['max_drawdown']:>10.2%}")
    print(f"    Calmar Ratio:       {metrics['calmar_ratio']:>10.4f}")


def save_csv(detail: pd.DataFrame, allocator_name: str, period_label: str, metrics: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    version_str = allocator_name.lower()
    filename = f"portfolio_{date_str}_{version_str}_{period_label}.csv"
    csv_path = os.path.join(OUTPUT_DIR, filename)

    detail.to_csv(csv_path, float_format='%.6f')
    print(f"\n  CSV saved: {csv_path}")

    metrics_path = os.path.join(OUTPUT_DIR, f"portfolio_{date_str}_{version_str}_{period_label}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Allocator: {allocator_name}\n")
        f.write(f"Period: {period_label}\n")
        f.write(f"Date: {date_str}\n\n")
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, float):
                    f.write(f"  {k}: {v:.6f}\n")
                else:
                    f.write(f"  {k}: {v}\n")
    print(f"  Metrics saved: {metrics_path}")

    return csv_path


def plot_nav(detail: pd.DataFrame, allocator_name: str, period_label: str, metrics: dict):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  [WARN] matplotlib not available, skipping chart")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    version_str = allocator_name.lower()

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [3, 1.5, 1.5]})

    ax1 = axes[0]
    ax1.plot(detail.index, detail['fund_cum_return'], label='All-Weather Portfolio', color='navy', linewidth=1.5)
    if detail['spy_bh_cum_return'].notna().any():
        ax1.plot(detail.index, detail['spy_bh_cum_return'], label='Buy & Hold SPY', color='gray', linewidth=1.0, alpha=0.7)
    ax1.set_title(f'All-Weather Portfolio ({allocator_name}) vs Buy&Hold SPY [{period_label}]', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))

    if metrics:
        text_lines = [
            f"Total Return: {metrics['total_return']:.2%}",
            f"Ann. Return: {metrics['annualized_return']:.2%}",
            f"Ann. Vol: {metrics['annualized_vol']:.2%}",
            f"Sharpe: {metrics['sharpe_ratio']:.4f}",
            f"Max DD: {metrics['max_drawdown']:.2%}",
            f"Calmar: {metrics['calmar_ratio']:.4f}",
        ]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.97, '\n'.join(text_lines), transform=ax1.transAxes,
                 fontsize=9, verticalalignment='top', bbox=props)

    ax2 = axes[1]
    weight_cols = [c for c in detail.columns if c.startswith('w_')]
    if weight_cols:
        labels = [c.replace('w_', '') for c in weight_cols]
        colors = ['#FFD700', '#4169E1', '#228B22', '#FF6600']
        data_stack = [detail[c].values for c in weight_cols]
        ax2.stackplot(detail.index.values, *data_stack, labels=labels, colors=colors[:len(weight_cols)], alpha=0.7)
    ax2.set_title('Asset Allocation Weights', fontsize=12)
    ax2.set_ylabel('Weight')
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='upper left', fontsize=9, ncol=4)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))

    ax3 = axes[2]
    running_max = detail['fund_cum_return'].cummax()
    drawdown = (detail['fund_cum_return'] - running_max) / running_max
    ax3.fill_between(detail.index, drawdown, 0, color='red', alpha=0.4)
    ax3.set_title('Drawdown', fontsize=12)
    ax3.set_ylabel('Drawdown')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, f"portfolio_{date_str}_{version_str}_{period_label}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {chart_path}")


def save_experiment_to_db(allocator_name, description, is_metrics, oos_metrics=None):
    from database.db_manager import insert_portfolio_experiment, update_portfolio_experiment_oos
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_id = f"{allocator_name}_{date_str}"

    insert_portfolio_experiment(
        exp_id=exp_id,
        allocator_name=allocator_name,
        description=description,
        is_sharpe=is_metrics.get('sharpe_ratio'),
        is_calmar=is_metrics.get('calmar_ratio'),
        is_return=is_metrics.get('annualized_return'),
        is_max_dd=is_metrics.get('max_drawdown'),
    )

    if oos_metrics:
        update_portfolio_experiment_oos(
            exp_id=exp_id,
            oos_sharpe=oos_metrics.get('sharpe_ratio'),
            oos_calmar=oos_metrics.get('calmar_ratio'),
            oos_return=oos_metrics.get('annualized_return'),
            oos_max_dd=oos_metrics.get('max_drawdown'),
        )

    return exp_id


def check_oos_gate(is_metrics: dict) -> bool:
    is_sharpe = is_metrics.get('sharpe_ratio', 0)
    is_calmar = is_metrics.get('calmar_ratio', 0)
    passed = is_sharpe > IS_SHARPE_GATE and is_calmar > IS_CALMAR_GATE
    return passed


def run_is_backtest(allocator, asset_data):
    print("\n" + "=" * 70)
    print("  >>> IS PERIOD BACKTEST (2007-2019) <<<")
    print("=" * 70)

    full_range, exposures, returns, prices, rf_series = align_data(asset_data, IS_START, IS_END)
    detail = run_backtest(allocator, full_range, exposures, returns, prices, rf_series)

    is_metrics = compute_metrics(detail['fund_daily_return'], detail['fund_cum_return'], label='IS Portfolio')
    print_metrics(is_metrics)

    if detail['spy_bh_cum_return'].notna().any():
        spy_ret = detail['ret_SPY'].reindex(detail.index).fillna(0.0)
        spy_cum = detail['spy_bh_cum_return']
        spy_metrics = compute_metrics(spy_ret, spy_cum, label='IS Buy&Hold SPY')
        print_metrics(spy_metrics)

    print(f"\n  Avg daily exposure: {detail['total_invested'].mean():.2%}")
    print(f"  Avg active assets:  {detail['active_assets'].mean():.1f}")
    print(f"  Avg cash portion:   {detail['cash_portion'].mean():.2%}")

    save_csv(detail, allocator.name, 'IS', is_metrics)
    plot_nav(detail, allocator.name, 'IS', is_metrics)

    return detail, is_metrics


def run_oos_backtest(allocator, asset_data):
    print("\n" + "=" * 70)
    print("  >>> OOS PERIOD BACKTEST (2020-2026) <<<")
    print("  *** THIS IS A BLIND TEST - RESULTS ARE FINAL ***")
    print("=" * 70)

    full_range, exposures, returns, prices, rf_series = align_data(asset_data, OOS_START, OOS_END)
    detail = run_backtest(allocator, full_range, exposures, returns, prices, rf_series)

    oos_metrics = compute_metrics(detail['fund_daily_return'], detail['fund_cum_return'], label='OOS Portfolio')
    print_metrics(oos_metrics)

    if detail['spy_bh_cum_return'].notna().any():
        spy_ret = detail['ret_SPY'].reindex(detail.index).fillna(0.0)
        spy_cum = detail['spy_bh_cum_return']
        spy_metrics = compute_metrics(spy_ret, spy_cum, label='OOS Buy&Hold SPY')
        print_metrics(spy_metrics)

    print(f"\n  Avg daily exposure: {detail['total_invested'].mean():.2%}")
    print(f"  Avg active assets:  {detail['active_assets'].mean():.1f}")
    print(f"  Avg cash portion:   {detail['cash_portion'].mean():.2%}")

    save_csv(detail, allocator.name, 'OOS', oos_metrics)
    plot_nav(detail, allocator.name, 'OOS', oos_metrics)

    return detail, oos_metrics


def run_full_backtest(allocator, asset_data):
    print("\n" + "=" * 70)
    print("  >>> FULL PERIOD BACKTEST (IS 2007-2019 + OOS 2020-2026) <<<")
    print("=" * 70)

    full_range, exposures, returns, prices, rf_series = align_data(asset_data, IS_START, OOS_END)
    detail = run_backtest(allocator, full_range, exposures, returns, prices, rf_series)

    is_mask = detail.index <= pd.Timestamp(IS_END)
    oos_mask = detail.index >= pd.Timestamp(OOS_START)

    is_daily_ret = detail.loc[is_mask, 'fund_daily_return']
    is_cum_ret = (1 + is_daily_ret.fillna(0.0)).cumprod()
    is_metrics = compute_metrics(is_daily_ret, is_cum_ret, label='IS Portfolio')

    oos_daily_ret = detail.loc[oos_mask, 'fund_daily_return']
    oos_cum_ret = (1 + oos_daily_ret.fillna(0.0)).cumprod()
    oos_metrics = compute_metrics(oos_daily_ret, oos_cum_ret, label='OOS Portfolio')

    full_daily_ret = detail['fund_daily_return']
    full_cum_ret = detail['fund_cum_return']
    full_metrics = compute_metrics(full_daily_ret, full_cum_ret, label='Full Period Portfolio')

    print_metrics(is_metrics)
    print_metrics(oos_metrics)
    print_metrics(full_metrics)

    if detail['spy_bh_cum_return'].notna().any():
        spy_ret = detail['ret_SPY'].reindex(detail.index).fillna(0.0)
        spy_is_daily = spy_ret[is_mask]
        spy_is_cum = (1 + spy_is_daily).cumprod()
        spy_oos_daily = spy_ret[oos_mask]
        spy_oos_cum = (1 + spy_oos_daily).cumprod()
        spy_full_cum = detail['spy_bh_cum_return']
        spy_is_metrics = compute_metrics(spy_is_daily, spy_is_cum, label='IS Buy&Hold SPY')
        spy_oos_metrics = compute_metrics(spy_oos_daily, spy_oos_cum, label='OOS Buy&Hold SPY')
        spy_full_metrics = compute_metrics(spy_ret, spy_full_cum, label='Full Buy&Hold SPY')
        print_metrics(spy_is_metrics)
        print_metrics(spy_oos_metrics)
        print_metrics(spy_full_metrics)

    print(f"\n  Avg daily exposure: {detail['total_invested'].mean():.2%}")
    print(f"  Avg active assets:  {detail['active_assets'].mean():.1f}")
    print(f"  Avg cash portion:   {detail['cash_portion'].mean():.2%}")

    save_csv(detail, allocator.name, 'FULL', full_metrics)
    plot_nav(detail, allocator.name, 'FULL', full_metrics)

    return detail, is_metrics, oos_metrics


def main():
    parser = argparse.ArgumentParser(description='All-Weather Portfolio Engine with IS/OOS Isolation')
    parser.add_argument('--allocator', type=str, default='V1DynamicEqualWeight',
                        help='Allocator class name (e.g. V1DynamicEqualWeight, V2RiskParity)')
    parser.add_argument('--period', type=str, default='is', choices=['is', 'oos', 'full'],
                        help='Backtest period: is (2007-2019), oos (2020-2026), full (IS+OOS)')
    parser.add_argument('--run-oos', action='store_true', default=False,
                        help='Force unlock OOS blind test (bypasses IS gate check)')
    parser.add_argument('--description', type=str, default='',
                        help='Description for this experiment run')
    parser.add_argument('--refresh-cache', action='store_true', default=False,
                        help='Force refresh all strategy caches with latest DB factors (use when factors have changed)')
    args = parser.parse_args()

    import portfolio_allocators.v1_dynamic_equal_weight
    import portfolio_allocators.v1_naive_equal_weight
    import portfolio_allocators.v2_dynamic_scalar_weight
    import portfolio_allocators.v3_risk_parity_allocator
    import portfolio_allocators.v4_risk_parity_with_caps
    import portfolio_allocators.v5_cross_asset_momentum
    import portfolio_allocators.v6_risk_parity_double_leverage
    import portfolio_allocators.v7_risk_parity_3x_nocap
    import portfolio_allocators.v8_risk_parity_3x_70cap
    from portfolio_allocators import get_allocator

    allocator_cls = get_allocator(args.allocator)
    allocator = allocator_cls()

    print("=" * 70)
    print(f"  All-Weather Portfolio Engine")
    print(f"  Allocator: {allocator.name}")
    print(f"  Description: {allocator.description}")
    print(f"  Period: {args.period}")
    print("=" * 70)

    asset_data = load_all_strategies(IS_START, OOS_END, refresh_cache=args.refresh_cache)

    if not asset_data:
        print("No data loaded!")
        return

    print(f"\n  Assets loaded: {list(asset_data.keys())}")

    from database.db_manager import init_db
    init_db()

    if args.period == 'is':
        detail, is_metrics = run_is_backtest(allocator, asset_data)

        gate_passed = check_oos_gate(is_metrics)
        print("\n" + "=" * 70)
        print(f"  IS Gate Check: Sharpe={is_metrics['sharpe_ratio']:.4f} (>{IS_SHARPE_GATE}), "
              f"Calmar={is_metrics['calmar_ratio']:.4f} (>{IS_CALMAR_GATE})")
        if gate_passed:
            print(f"  *** IS GATE PASSED *** You may proceed to OOS blind test.")
            print(f"  Run: python portfolio_engine.py --allocator {args.allocator} --period oos --run-oos")
        else:
            print(f"  *** IS GATE NOT PASSED *** OOS is locked. Improve the allocator first.")
        print("=" * 70)

        save_experiment_to_db(allocator.name, args.description or allocator.description, is_metrics)

    elif args.period == 'oos':
        is_detail, is_metrics = run_is_backtest(allocator, asset_data)

        gate_passed = check_oos_gate(is_metrics)
        force_unlock = args.run_oos

        print("\n" + "=" * 70)
        print(f"  IS Gate Check: Sharpe={is_metrics['sharpe_ratio']:.4f} (>{IS_SHARPE_GATE}), "
              f"Calmar={is_metrics['calmar_ratio']:.4f} (>{IS_CALMAR_GATE})")
        print(f"  Gate passed: {gate_passed}, Force unlock (--run-oos): {force_unlock}")

        if not gate_passed and not force_unlock:
            print(f"\n  *** OOS IS LOCKED *** IS metrics do not meet the gate threshold.")
            print(f"  To force unlock, use: --run-oos flag")
            print("=" * 70)
            save_experiment_to_db(allocator.name, args.description or allocator.description, is_metrics)
            return

        if gate_passed and not force_unlock:
            print(f"\n  IS gate passed, but --run-oos flag not provided.")
            print(f"  To proceed with OOS blind test, re-run with --run-oos flag.")
            print("=" * 70)
            save_experiment_to_db(allocator.name, args.description or allocator.description, is_metrics)
            return

        print(f"\n  >>> OOS UNLOCKED. Proceeding with blind test... <<<")
        print("=" * 70)

        oos_detail, oos_metrics = run_oos_backtest(allocator, asset_data)

        exp_id = save_experiment_to_db(
            allocator.name, args.description or allocator.description,
            is_metrics, oos_metrics
        )
        print(f"\n  Experiment saved: {exp_id}")

    elif args.period == 'full':
        detail, is_metrics, oos_metrics = run_full_backtest(allocator, asset_data)

        save_experiment_to_db(
            allocator.name, args.description or allocator.description,
            is_metrics, oos_metrics
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
