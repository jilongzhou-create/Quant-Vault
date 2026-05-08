#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
GCUSD 宏观估值模型 - IS 校准脚本

目标: 在 IS 周期内建立黄金的宏观估值模型
  Y = ln(GCUSD)
  X = [Constant, DFII10, ln(DTWEXBGS)]

数据来源:
  - market_data_gold: GCUSD 日线 OHLCV
  - factor_data (symbol='MACRO'): DFII10 (10Y TIPS), DTWEXBGS (Trade Weighted USD Index)

注意:
  - factor_data 中存储的是经过 30 日滚动 Z-Score 标准化后的值，不是原始水平值
  - 本脚本从 raw_data 中重新提取原始水平值用于回归
  - 若 raw_data 中无足够历史数据，则回退使用 factor_data 中的 Z-Score 值
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DB_PATH
from database.db_manager import get_raw_data_by_source
from isolated_macro.db_schema_macro import ALL_MACRO_DDL

IS_START = '2007-01-01'
IS_END = '2019-12-31'


def load_gold_prices():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT timestamp, close FROM market_data_gold WHERE symbol = 'GCUSD' ORDER BY timestamp",
        conn
    )
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize()
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
    df.rename(columns={'close': 'gold_close'}, inplace=True)
    print(f"[DATA] Gold prices: {len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}")
    return df


def load_fred_raw(series_id):
    source_id = f'fred_{series_id}'
    raw_records = get_raw_data_by_source(source_id)
    if not raw_records:
        print(f"[WARN] No raw_data for {source_id}")
        return pd.DataFrame()

    rows = []
    for rec in raw_records:
        try:
            raw_content = rec.get('raw_content', rec) if isinstance(rec, dict) else json.loads(rec[4]) if len(rec) > 4 else {}
            if isinstance(raw_content, str):
                raw_content = json.loads(raw_content)
            value_str = raw_content.get('value', '.')
            event_ts = raw_content.get('date', rec.get('event_timestamp', ''))
            if value_str and value_str != '.':
                rows.append({'timestamp': event_ts, 'value': float(value_str)})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize()
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
    df.rename(columns={'value': series_id.lower()}, inplace=True)
    print(f"[DATA] FRED raw {series_id}: {len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}")
    return df


def load_fred_factor(factor_name):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT timestamp, factor_value FROM factor_data WHERE symbol='MACRO' AND factor_name = ? ORDER BY timestamp",
        conn,
        params=(factor_name,)
    )
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize()
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
    df.rename(columns={'factor_value': factor_name}, inplace=True)
    print(f"[DATA] FRED factor {factor_name}: {len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}")
    return df


def ols_regression(Y, X_df):
    """
    OLS regression with full statistical output.
    Y: Series (dependent variable)
    X_df: DataFrame (independent variables, no constant)
    Returns: dict with coefficients, statistics, and summary string
    """
    X_with_const = np.column_stack([np.ones(len(X_df)), X_df.values])
    col_names = ['const'] + list(X_df.columns)
    y = Y.values

    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]

    y_hat = X_with_const @ beta
    resid = y - y_hat
    n = len(y)
    k = X_with_const.shape[1]

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_reg = ss_tot - ss_res

    r_squared = 1 - ss_res / ss_tot
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)

    mse = ss_res / (n - k)
    var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const)
    se_beta = np.sqrt(np.diag(var_beta))
    t_stats = beta / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

    f_stat = (ss_reg / (k - 1)) / mse if k > 1 else np.nan
    f_p_value = 1 - stats.f.cdf(f_stat, k - 1, n - k) if k > 1 else np.nan

    aic = n * np.log(ss_res / n) + 2 * k
    bic = n * np.log(ss_res / n) + k * np.log(n)

    dw = np.sum(np.diff(resid) ** 2) / ss_res if n > 1 else np.nan

    summary_lines = []
    summary_lines.append("=" * 78)
    summary_lines.append("                         OLS Regression Results")
    summary_lines.append("=" * 78)
    summary_lines.append(f"Dep. Variable:          {Y.name}")
    summary_lines.append(f"No. Observations:       {n}")
    summary_lines.append(f"Df Model:               {k - 1}")
    summary_lines.append(f"Df Residuals:           {n - k}")
    summary_lines.append(f"R-squared:              {r_squared:.6f}")
    summary_lines.append(f"Adj. R-squared:         {adj_r_squared:.6f}")
    summary_lines.append(f"F-statistic:            {f_stat:.4f}")
    summary_lines.append(f"Prob (F-statistic):     {f_p_value:.6e}")
    summary_lines.append(f"AIC:                    {aic:.4f}")
    summary_lines.append(f"BIC:                    {bic:.4f}")
    summary_lines.append(f"Durbin-Watson:          {dw:.4f}")
    summary_lines.append("=" * 78)
    summary_lines.append("")
    header = f"{'':>20s} {'coef':>12s} {'std err':>12s} {'t':>10s} {'P>|t|':>10s} {'[0.025':>10s} {'0.975]':>10s}"
    summary_lines.append(header)
    summary_lines.append("-" * 78)

    for i, name in enumerate(col_names):
        ci_low = beta[i] - stats.t.ppf(0.975, n - k) * se_beta[i]
        ci_high = beta[i] + stats.t.ppf(0.975, n - k) * se_beta[i]
        sig = ""
        if p_values[i] < 0.001:
            sig = "***"
        elif p_values[i] < 0.01:
            sig = "**"
        elif p_values[i] < 0.05:
            sig = "*"
        elif p_values[i] < 0.1:
            sig = "."
        line = f"{name:>20s} {beta[i]:>12.6f} {se_beta[i]:>12.6f} {t_stats[i]:>10.4f} {p_values[i]:>10.6f} {ci_low:>10.4f} {ci_high:>10.4f} {sig}"
        summary_lines.append(line)

    summary_lines.append("-" * 78)
    summary_lines.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    summary_lines.append("=" * 78)

    summary = "\n".join(summary_lines)

    coefficients = {}
    for i, name in enumerate(col_names):
        coefficients[name] = {
            'coef': float(beta[i]),
            'std_err': float(se_beta[i]),
            't_stat': float(t_stats[i]),
            'p_value': float(p_values[i])
        }

    return {
        'summary': summary,
        'coefficients': coefficients,
        'r_squared': float(r_squared),
        'adj_r_squared': float(adj_r_squared),
        'f_statistic': float(f_stat),
        'f_p_value': float(f_p_value),
        'durbin_watson': float(dw),
        'n_obs': n,
        'residuals': resid,
        'y_hat': y_hat,
    }


def save_model_to_registry(model_id, target_symbol, formula_desc, params_json, is_start, is_end):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO macro_model_registry
        (model_id, target_symbol, formula_desc, params_json, is_start_date, is_end_date)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (model_id, target_symbol, formula_desc, params_json, is_start, is_end))
    conn.commit()
    conn.close()
    print(f"\n[DB] Model '{model_id}' saved to macro_model_registry")


def main():
    print("=" * 78)
    print("GCUSD Macro Valuation Model - IS Calibration")
    print(f"IS Period: {IS_START} ~ {IS_END}")
    print("=" * 78)

    # Step 1: Load gold prices
    print("\n[Step 1] Loading GCUSD prices from market_data_gold...")
    df_gold = load_gold_prices()
    if df_gold.empty:
        print("ERROR: No gold price data available!")
        return

    # Step 2: Load macro factors - try raw_data first, fallback to factor_data
    print("\n[Step 2] Loading macro factors...")

    df_dfii10_raw = load_fred_raw('DFII10')
    df_dtwexbgs_raw = load_fred_raw('DTWEXBGS')

    use_raw = False
    if not df_dfii10_raw.empty and not df_dtwexbgs_raw.empty:
        dfii10_start = df_dfii10_raw.index[0]
        dtwexbgs_start = df_dtwexbgs_raw.index[0]
        earliest = max(dfii10_start, dtwexbgs_start)
        is_start_ts = pd.Timestamp(IS_START)
        if earliest <= is_start_ts + pd.Timedelta(days=5):
            print(f"[INFO] Raw data covers IS period (earliest: {earliest.date()}, IS start: {IS_START})")
            use_raw = True
        else:
            print(f"[WARN] Raw data starts at {earliest.date()}, significantly after IS start {IS_START}")

    if use_raw:
        df_dfii10 = df_dfii10_raw
        df_dtwexbgs = df_dtwexbgs_raw
        dfii10_col = 'dfii10'
        dtwexbgs_col = 'dtwexbgs'
        print("[INFO] Using RAW (original level) values from raw_data")
    else:
        print("[WARN] Raw data insufficient for IS period. Falling back to factor_data (Z-Score values).")
        print("[WARN] NOTE: Z-Score values are standardized, not original levels.")
        print("[WARN] Regression coefficients will be on Z-Score scale, not economic level scale.")
        df_dfii10 = load_fred_factor('dfii10')
        df_dtwexbgs = load_fred_factor('dtwexbgs')
        dfii10_col = 'dfii10'
        dtwexbgs_col = 'dtwexbgs'

    if df_dfii10.empty or df_dtwexbgs.empty:
        print("ERROR: Missing macro factor data!")
        return

    # Step 3: Merge and process
    print("\n[Step 3] Merging and processing data...")

    df = df_gold.copy()
    df = df.join(df_dfii10[[dfii10_col]], how='inner')
    df = df.join(df_dtwexbgs[[dtwexbgs_col]], how='inner')

    df = df.sort_index()
    df = df.ffill()
    df = df.dropna()

    print(f"[DATA] After merge + ffill + dropna: {len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}")

    if len(df) < 30:
        print(f"ERROR: Only {len(df)} rows after merge, insufficient for regression!")
        return

    # Step 4: Transformations
    print("\n[Step 4] Applying transformations...")
    df['ln_gold'] = np.log(df['gold_close'])

    if use_raw:
        df['ln_dxy'] = np.log(df[dtwexbgs_col])
        print(f"  ln_gold = Log(gold_close)")
        print(f"  DFII10 = raw TIPS yield (original scale)")
        print(f"  ln_dxy = Log(DTWEXBGS) (original scale)")
    else:
        df['ln_dxy'] = df[dtwexbgs_col]
        print(f"  ln_gold = Log(gold_close)")
        print(f"  DFII10 = Z-Score (standardized)")
        print(f"  ln_dxy = Z-Score of DTWEXBGS (standardized, NOT log)")

    # Step 5: Filter IS period
    print(f"\n[Step 5] Filtering IS period: {IS_START} ~ {IS_END}...")
    is_mask = (df.index >= IS_START) & (df.index <= IS_END)
    df_is = df.loc[is_mask].copy()
    print(f"[DATA] IS period rows: {len(df_is)}")

    if len(df_is) < 30:
        print(f"ERROR: Only {len(df_is)} rows in IS period, insufficient for regression!")
        print("Attempting to use all available data for calibration...")
        df_is = df.copy()
        print(f"[DATA] Using all available data: {len(df_is)} rows, {df_is.index[0].date()} ~ {df_is.index[-1].date()}")

    # Step 6: OLS Regression
    print("\n[Step 6] Running OLS regression...")
    print(f"  Y = ln_gold")
    print(f"  X = [Constant, DFII10, ln_dxy]")

    Y = df_is['ln_gold']
    X = df_is[['dfii10', 'ln_dxy']]

    result = ols_regression(Y, X)

    print("\n" + result['summary'])

    # Step 7: Economic interpretation
    print("\n[Step 7] Economic interpretation:")
    coef = result['coefficients']
    b0 = coef['const']['coef']
    b1 = coef['dfii10']['coef']
    b2 = coef['ln_dxy']['coef']

    print(f"  Intercept (b0) = {b0:.6f}")
    print(f"  b_DFII10   (b1) = {b1:.6f}  {'[SIGNIFICANT]' if coef['dfii10']['p_value'] < 0.05 else '[NOT SIGNIFICANT]'}")
    print(f"  b_lnDXY    (b2) = {b2:.6f}  {'[SIGNIFICANT]' if coef['ln_dxy']['p_value'] < 0.05 else '[NOT SIGNIFICANT]'}")

    if use_raw:
        if b1 < 0:
            print(f"  -> b_DFII10 < 0: TIPS yield up => Gold down (expected: gold is anti-real-rate)")
        else:
            print(f"  -> b_DFII10 > 0: UNEXPECTED sign! Gold should fall when real rates rise.")
        if b2 < 0:
            print(f"  -> b_lnDXY < 0: DXY up => Gold down (expected: gold priced in USD)")
        else:
            print(f"  -> b_lnDXY > 0: UNEXPECTED sign! Gold should fall when USD strengthens.")
    else:
        print(f"  [NOTE] Using Z-Score values, economic sign interpretation is on standardized scale.")

    print(f"\n  R-squared = {result['r_squared']:.4f}  ({result['r_squared']*100:.1f}% variance explained)")
    print(f"  Adj R-sq  = {result['adj_r_squared']:.4f}")
    print(f"  F-stat    = {result['f_statistic']:.4f}  (p = {result['f_p_value']:.2e})")
    print(f"  Durbin-Watson = {result['durbin_watson']:.4f}  ({'positive autocorrelation' if result['durbin_watson'] < 1.5 else 'near no autocorrelation' if result['durbin_watson'] < 2.5 else 'negative autocorrelation'})")

    # Step 8: Save to database
    print("\n[Step 8] Saving model to macro_model_registry...")

    model_id = 'gold_macro_v1'
    target_symbol = 'GCUSD'

    if use_raw:
        formula_desc = f"ln(GCUSD) = {b0:.6f} + {b1:.6f}*DFII10 + {b2:.6f}*ln(DTWEXBGS)"
    else:
        formula_desc = f"ln(GCUSD) = {b0:.6f} + {b1:.6f}*Z(DFII10) + {b2:.6f}*Z(DTWEXBGS)"

    params = {
        "intercept": round(b0, 8),
        "b_DFII10": round(b1, 8),
        "b_lnDXY": round(b2, 8),
        "r_squared": round(result['r_squared'], 6),
        "adj_r_squared": round(result['adj_r_squared'], 6),
        "f_statistic": round(result['f_statistic'], 4),
        "f_p_value": result['f_p_value'],
        "durbin_watson": round(result['durbin_watson'], 4),
        "n_obs": result['n_obs'],
        "data_mode": "raw_level" if use_raw else "zscore",
        "coef_const_se": round(coef['const']['std_err'], 8),
        "coef_const_t": round(coef['const']['t_stat'], 4),
        "coef_const_p": coef['const']['p_value'],
        "coef_dfii10_se": round(coef['dfii10']['std_err'], 8),
        "coef_dfii10_t": round(coef['dfii10']['t_stat'], 4),
        "coef_dfii10_p": coef['dfii10']['p_value'],
        "coef_ln_dxy_se": round(coef['ln_dxy']['std_err'], 8),
        "coef_ln_dxy_t": round(coef['ln_dxy']['t_stat'], 4),
        "coef_ln_dxy_p": coef['ln_dxy']['p_value'],
    }
    params_json = json.dumps(params, indent=2)

    actual_is_start = str(df_is.index[0].date())
    actual_is_end = str(df_is.index[-1].date())

    save_model_to_registry(
        model_id=model_id,
        target_symbol=target_symbol,
        formula_desc=formula_desc,
        params_json=params_json,
        is_start=actual_is_start,
        is_end=actual_is_end
    )

    print(f"\n  model_id     = {model_id}")
    print(f"  target_symbol= {target_symbol}")
    print(f"  formula      = {formula_desc}")
    print(f"  IS period    = {actual_is_start} ~ {actual_is_end}")
    print(f"  params_json  =")
    print(params_json)

    print("\n" + "=" * 78)
    print("IS Calibration Complete!")
    print("=" * 78)


if __name__ == '__main__':
    main()
