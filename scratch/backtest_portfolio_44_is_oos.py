#!/usr/bin/env python3
"""
Portfolio 44 IS/OOS 精确分析
IS = 2021.05 ~ 2026.04 (策略开发期)
OOS = 2018.01 ~ 2021.04 (样本外验证期)
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
from trading_engine.backtest_engine import load_daily_data_directly, compile_strategy
from database.db_manager import DB_PATH

PORTFOLIO_ID = 44
COST_RATE = 0.0005
IS_START = '2021-05-01'
IS_END = '2026-04-30'
OOS_START = '2018-01-01'
OOS_END = '2021-04-30'


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
        strategies.append({'dir_id': dir_id, 'code': code, 'params': params, 'name': name})
    return strategies


def generate_signals(strategy, df):
    try:
        func = compile_strategy(strategy['code'])
        signals = func(df.copy(), strategy['params'])
        if isinstance(signals, pd.Series):
            return signals.reindex(df.index).fillna(0)
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

    return raw_sum.apply(scale_signal).clip(-1, 1)


def run_backtest_period(df, signal, start, end):
    mask = (df.index >= start) & (df.index <= end + ' 23:59:59')
    df_sub = df[mask].copy()
    sig_sub = signal[mask].copy()
    if len(df_sub) < 10:
        return None

    price_return = df_sub['close'].pct_change().fillna(0)
    position = sig_sub.shift(1).fillna(0)
    turnover = position.diff().fillna(0).abs()
    fee = turnover * COST_RATE
    daily_return = position * price_return - fee
    nav = (1 + daily_return).cumprod()
    nav.iloc[0] = 1.0

    return {'df': df_sub, 'signal': sig_sub, 'daily_return': daily_return,
            'nav': nav, 'position': position, 'price_return': price_return}


def calc_metrics(result, label=""):
    if result is None:
        return None
    nav = result['nav']
    dr = result['daily_return']
    n = len(nav)
    if n < 2:
        return None

    total_ret = nav.iloc[-1] - 1
    years = n / 365.25
    annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    peak = nav.expanding(min_periods=1).max()
    dd = (nav - peak) / peak
    max_dd = dd.min()
    std = dr.std()
    sharpe = (dr.mean() / std) * np.sqrt(365.25) if std > 1e-9 else 0
    calmar = annual_ret / abs(max_dd) if abs(max_dd) > 1e-9 else 0
    win_rate = (dr > 0).sum() / len(dr[dr != 0]) if len(dr[dr != 0]) > 0 else 0

    return {'label': label, 'start': nav.index[0].strftime('%Y-%m-%d'),
            'end': nav.index[-1].strftime('%Y-%m-%d'), 'days': n,
            'total_return': total_ret, 'annual_return': annual_ret,
            'sharpe': sharpe, 'max_dd': max_dd, 'calmar': calmar,
            'win_rate': win_rate, 'vol': std * np.sqrt(365.25)}


def calc_yearly(result):
    if result is None:
        return []
    dr = result['daily_return']
    nav = result['nav']
    pos = result['position']
    data = []
    for year, grp in dr.groupby(dr.index.year):
        yr_ret = (1 + grp).prod() - 1
        yr_vol = grp.std() * np.sqrt(365.25)
        yr_sharpe = (grp.mean() / grp.std()) * np.sqrt(365.25) if grp.std() > 1e-9 else 0
        nav_grp = nav[nav.index.year == year]
        peak = nav_grp.expanding(min_periods=1).max()
        yr_dd = ((nav_grp - peak) / peak).min()
        yr_calmar = yr_ret / abs(yr_dd) if abs(yr_dd) > 1e-9 else 0
        avg_pos = pos[pos.index.year == year].mean()
        data.append({'year': year, 'return': yr_ret, 'vol': yr_vol,
                     'sharpe': yr_sharpe, 'max_dd': yr_dd, 'calmar': yr_calmar,
                     'avg_position': avg_pos})
    return data


def main():
    print("=" * 80)
    print("  Portfolio 44 IS/OOS Analysis")
    print(f"  OOS (验证期): {OOS_START} ~ {OOS_END}")
    print(f"  IS  (开发期): {IS_START} ~ {IS_END}")
    print("=" * 80)

    strategies = load_portfolio_strategies()
    print(f"\n  Loaded {len(strategies)} strategies")

    df = load_daily_data_directly(symbol='BTC_USDT', target_asset='crypto')
    df = df[df.index >= OOS_START]
    df = df[df.index <= IS_END + ' 23:59:59']
    print(f"  Data: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}, {len(df)} days")

    signals_dict = {}
    for s in strategies:
        sig = generate_signals(s, df)
        signals_dict[s['name']] = sig

    signals_df = pd.DataFrame(signals_dict, index=df.index)
    combined = combine_signals_scaling(signals_df)

    # === OOS Backtest ===
    print("\n" + "#" * 80)
    print(f"  OOS (样本外验证): {OOS_START} ~ {OOS_END}")
    print("#" * 80)

    result_oos = run_backtest_period(df, combined, OOS_START, OOS_END)
    m_oos = calc_metrics(result_oos, "OOS")

    result_bh_oos = run_backtest_period(df, pd.Series(1.0, index=df.index), OOS_START, OOS_END)
    m_bh_oos = calc_metrics(result_bh_oos, "B&H OOS")

    print(f"\n  {'Metric':<22s}  {'Portfolio44':>12s}  {'Buy&Hold':>12s}  {'Alpha':>12s}")
    print("  " + "-" * 62)
    if m_oos and m_bh_oos:
        for label, key, fmt in [
            ('Total Return', 'total_return', '%'),
            ('Annual Return', 'annual_return', '%'),
            ('Volatility', 'vol', '%'),
            ('Sharpe Ratio', 'sharpe', ''),
            ('Max Drawdown', 'max_dd', '%'),
            ('Calmar Ratio', 'calmar', ''),
            ('Win Rate', 'win_rate', '%'),
        ]:
            v = m_oos[key]
            vbh = m_bh_oos[key]
            alpha = v - vbh
            if fmt == '%':
                print(f"  {label:<22s}  {v:>11.2%}  {vbh:>11.2%}  {alpha:>+11.2%}")
            else:
                print(f"  {label:<22s}  {v:>12.4f}  {vbh:>12.4f}  {alpha:>+12.4f}")

    # OOS Yearly
    print(f"\n  OOS Yearly Breakdown:")
    print(f"  {'Year':>6s}  {'Return':>10s}  {'Sharpe':>8s}  {'MaxDD':>10s}  {'Calmar':>8s}  {'AvgPos':>8s}  {'B&H Ret':>10s}  {'B&H Sharpe':>10s}")
    print("  " + "-" * 80)
    yearly_oos = calc_yearly(result_oos)
    yearly_bh_oos = calc_yearly(result_bh_oos)
    bh_map = {y['year']: y for y in yearly_bh_oos}
    for y in yearly_oos:
        bh_y = bh_map.get(y['year'], {})
        bh_ret = bh_y.get('return', 0)
        bh_sharpe = bh_y.get('sharpe', 0)
        print(f"  {y['year']:>6d}  {y['return']:>10.2%}  {y['sharpe']:>8.2f}  {y['max_dd']:>10.2%}  {y['calmar']:>8.2f}  {y['avg_position']:>8.4f}  {bh_ret:>10.2%}  {bh_sharpe:>10.2f}")

    # === IS Backtest ===
    print("\n" + "#" * 80)
    print(f"  IS (样本内开发): {IS_START} ~ {IS_END}")
    print("#" * 80)

    result_is = run_backtest_period(df, combined, IS_START, IS_END)
    m_is = calc_metrics(result_is, "IS")

    result_bh_is = run_backtest_period(df, pd.Series(1.0, index=df.index), IS_START, IS_END)
    m_bh_is = calc_metrics(result_bh_is, "B&H IS")

    print(f"\n  {'Metric':<22s}  {'Portfolio44':>12s}  {'Buy&Hold':>12s}  {'Alpha':>12s}")
    print("  " + "-" * 62)
    if m_is and m_bh_is:
        for label, key, fmt in [
            ('Total Return', 'total_return', '%'),
            ('Annual Return', 'annual_return', '%'),
            ('Volatility', 'vol', '%'),
            ('Sharpe Ratio', 'sharpe', ''),
            ('Max Drawdown', 'max_dd', '%'),
            ('Calmar Ratio', 'calmar', ''),
            ('Win Rate', 'win_rate', '%'),
        ]:
            v = m_is[key]
            vbh = m_bh_is[key]
            alpha = v - vbh
            if fmt == '%':
                print(f"  {label:<22s}  {v:>11.2%}  {vbh:>11.2%}  {alpha:>+11.2%}")
            else:
                print(f"  {label:<22s}  {v:>12.4f}  {vbh:>12.4f}  {alpha:>+12.4f}")

    # IS Yearly
    print(f"\n  IS Yearly Breakdown:")
    print(f"  {'Year':>6s}  {'Return':>10s}  {'Sharpe':>8s}  {'MaxDD':>10s}  {'Calmar':>8s}  {'AvgPos':>8s}  {'B&H Ret':>10s}  {'B&H Sharpe':>10s}")
    print("  " + "-" * 80)
    yearly_is = calc_yearly(result_is)
    yearly_bh_is = calc_yearly(result_bh_is)
    bh_map2 = {y['year']: y for y in yearly_bh_is}
    for y in yearly_is:
        bh_y = bh_map2.get(y['year'], {})
        bh_ret = bh_y.get('return', 0)
        bh_sharpe = bh_y.get('sharpe', 0)
        print(f"  {y['year']:>6d}  {y['return']:>10.2%}  {y['sharpe']:>8.2f}  {y['max_dd']:>10.2%}  {y['calmar']:>8.2f}  {y['avg_position']:>8.4f}  {bh_ret:>10.2%}  {bh_sharpe:>10.2f}")

    # === IS vs OOS Degradation ===
    print("\n" + "=" * 80)
    print("  IS vs OOS DEGRADATION ANALYSIS")
    print("=" * 80)
    if m_is and m_oos:
        print(f"\n  {'Metric':<22s}  {'IS':>12s}  {'OOS':>12s}  {'Degradation':>12s}  {'Verdict':>10s}")
        print("  " + "-" * 72)
        for label, key, fmt, threshold in [
            ('Sharpe Ratio', 'sharpe', '', 0.5),
            ('Calmar Ratio', 'calmar', '', 0.5),
            ('Annual Return', 'annual_return', '%', None),
            ('Max Drawdown', 'max_dd', '%', None),
            ('Win Rate', 'win_rate', '%', None),
        ]:
            v_is = m_is[key]
            v_oos = m_oos[key]
            if threshold and abs(v_is) > 1e-9:
                degradation = (v_is - v_oos) / abs(v_is)
                verdict = "OK" if degradation < threshold else "WARNING"
                if fmt == '%':
                    print(f"  {label:<22s}  {v_is:>11.2%}  {v_oos:>11.2%}  {degradation:>+11.1%}  {verdict:>10s}")
                else:
                    print(f"  {label:<22s}  {v_is:>12.4f}  {v_oos:>12.4f}  {degradation:>+11.1%}  {verdict:>10s}")
            else:
                if fmt == '%':
                    print(f"  {label:<22s}  {v_is:>11.2%}  {v_oos:>11.2%}  {'N/A':>12s}  {'':>10s}")
                else:
                    print(f"  {label:<22s}  {v_is:>12.4f}  {v_oos:>12.4f}  {'N/A':>12s}  {'':>10s}")

        # Key ratio
        sharpe_is = m_is['sharpe']
        sharpe_oos = m_oos['sharpe']
        if abs(sharpe_is) > 1e-9:
            sharpe_ratio = sharpe_oos / sharpe_is
            print(f"\n  OOS/IS Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"  (Industry rule of thumb: >0.5 is acceptable, >0.7 is good)")

    # === Individual strategy OOS analysis ===
    print("\n" + "=" * 80)
    print("  INDIVIDUAL STRATEGY: IS vs OOS Sharpe")
    print("=" * 80)
    print(f"  {'Strategy':<40s}  {'IS Sharpe':>10s}  {'OOS Sharpe':>10s}  {'OOS/IS':>8s}")
    print("  " + "-" * 72)

    for name, sig in signals_dict.items():
        r_is = run_backtest_period(df, sig, IS_START, IS_END)
        r_oos = run_backtest_period(df, sig, OOS_START, OOS_END)
        m_is_s = calc_metrics(r_is)
        m_oos_s = calc_metrics(r_oos)
        is_sharpe = m_is_s['sharpe'] if m_is_s else 0
        oos_sharpe = m_oos_s['sharpe'] if m_oos_s else 0
        ratio = oos_sharpe / is_sharpe if abs(is_sharpe) > 1e-9 else 0
        print(f"  {name:<40s}  {is_sharpe:>10.2f}  {oos_sharpe:>10.2f}  {ratio:>8.2f}")

    print("\n" + "=" * 80)
    print("  Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
