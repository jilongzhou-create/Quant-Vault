#!/usr/bin/env python3
"""
V8 数据工程 - 宏观基础数据池扩充脚本

获取并落库以下数据:
  1. FRED DEXCHUS (USD/CNY 汇率, 日频)
  2. AkShare SGE Au99.99 (上海金交所现货黄金, 日频, CNY/克)
  3. 派生: SGE Premium (上海金溢价, USD/oz)

数据存储: factor_data 表 (symbol='MACRO')
Lookahead Bias 防护: 所有数据加 lag 后作为时间戳
  - FRED 日频: +1 天
  - AkShare 日频: +1 天 (T+1 才可获取)
  - 派生指标: 取各输入序列的最大 lag
"""

import sys
import os
import time
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DB_PATH
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.environ.get("FRED_API_KEY")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

GRAMS_PER_TROY_OZ = 31.1035


def _create_session():
    from data_pipeline.adapters.macro_fred_adapter import create_retry_session
    return create_retry_session()


def _save_factor_data(df_factor):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    saved = 0
    for _, row in df_factor.iterrows():
        try:
            ts = row['timestamp']
            if hasattr(ts, 'strftime'):
                ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
            else:
                ts_str = str(ts)
            cursor.execute(
                "INSERT OR IGNORE INTO factor_data (symbol, timestamp, factor_name, factor_value) VALUES (?, ?, ?, ?)",
                ('MACRO', ts_str, row['factor_name'], float(row['factor_value']))
            )
            if cursor.rowcount > 0:
                saved += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    return saved


def _save_factor_metadata(factor_name, description, source, unit='', update_freq=''):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO factor_metadata (factor_name, symbol, description, source, unit, update_freq) VALUES (?, ?, ?, ?, ?, ?)",
        (factor_name, 'MACRO', description, source, unit, update_freq)
    )
    conn.commit()
    conn.close()


def _get_latest_factor_timestamp(factor_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT MAX(timestamp) FROM factor_data WHERE symbol='MACRO' AND factor_name=?",
        (factor_name,)
    )
    row = cursor.fetchone()
    conn.close()
    if row and row[0]:
        return pd.Timestamp(row[0])
    return None


def fetch_fred_dexchus(start_date='2007-01-01'):
    """
    从 FRED 获取 DEXCHUS (USD/CNY 汇率) 并存入 factor_data

    DEXCHUS 是美联储发布的美元兑人民币汇率（日频），
    用于计算上海金溢价的核心输入。
    """
    print("\n" + "=" * 60)
    print("  [1/3] Fetching FRED DEXCHUS (USD/CNY Exchange Rate)")
    print("=" * 60)

    if not FRED_API_KEY:
        print("[ERROR] FRED_API_KEY not configured!")
        return 0

    factor_name = 'dexchus'
    latest_ts = _get_latest_factor_timestamp(factor_name)
    if latest_ts:
        obs_start = (latest_ts - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"  Incremental from: {obs_start}")
    else:
        obs_start = start_date
        print(f"  Full sync from: {obs_start}")

    session = _create_session()

    try:
        params = {
            'series_id': 'DEXCHUS',
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'observation_start': obs_start,
        }
        resp = session.get(FRED_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(0.6)

        data = resp.json()
        observations = data.get('observations', [])
        if not observations:
            print("  No observations returned!")
            return 0

        rows = []
        for obs in observations:
            val = obs.get('value', '.')
            if val and val != '.':
                event_date = pd.Timestamp(obs['date'])
                lagged_date = event_date + pd.Timedelta(days=1)
                rows.append({
                    'timestamp': lagged_date,
                    'factor_name': factor_name,
                    'factor_value': float(val),
                })

        if not rows:
            print("  No valid observations!")
            return 0

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=['timestamp', 'factor_name']).sort_values('timestamp')
        saved = _save_factor_data(df)

        _save_factor_metadata(
            factor_name,
            'USD/CNY Exchange Rate (FRED DEXCHUS)',
            'FRED',
            'CNY per USD',
            'D'
        )

        print(f"  DEXCHUS: {len(rows)} observations, {saved} new rows saved")
        print(f"  Range: {df['timestamp'].iloc[0].date()} ~ {df['timestamp'].iloc[-1].date()}")
        return saved

    except Exception as e:
        print(f"  [ERROR] DEXCHUS fetch failed: {e}")
        return 0


def fetch_sge_au9999(start_date='2007-01-01'):
    """
    从 AkShare 获取上海黄金交易所 Au99.99 日线数据并存入 factor_data

    使用 ak.spot_hist_sge(symbol="Au99.99") 获取，
    存储收盘价（CNY/克），加 1 天 lag 防止 lookahead bias。
    """
    print("\n" + "=" * 60)
    print("  [2/3] Fetching SGE Au99.99 (Shanghai Gold Exchange)")
    print("=" * 60)

    try:
        import akshare as ak
    except ImportError:
        print("[ERROR] akshare not installed! Run: pip install akshare")
        return 0

    factor_name = 'sge_au9999'

    try:
        df = ak.spot_hist_sge(symbol="Au99.99")
        if df.empty:
            print("  No data returned from AkShare!")
            return 0

        print(f"  AkShare returned {len(df)} rows")

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        latest_ts = _get_latest_factor_timestamp(factor_name)
        if latest_ts:
            latest_date = latest_ts - pd.Timedelta(days=1)
            df = df[df['date'] > latest_date]
            print(f"  Incremental from: {latest_date.date()}")

        if df.empty:
            print("  No new data after incremental filter!")
            return 0

        rows = []
        for _, row in df.iterrows():
            event_date = row['date']
            lagged_date = event_date + pd.Timedelta(days=1)
            rows.append({
                'timestamp': lagged_date,
                'factor_name': factor_name,
                'factor_value': float(row['close']),
            })

        df_factor = pd.DataFrame(rows)
        df_factor = df_factor.drop_duplicates(subset=['timestamp', 'factor_name']).sort_values('timestamp')
        saved = _save_factor_data(df_factor)

        _save_factor_metadata(
            factor_name,
            'SGE Au99.99 Close Price (CNY/g, Shanghai Gold Exchange)',
            'AkShare',
            'CNY/g',
            'D'
        )

        print(f"  SGE Au99.99: {len(rows)} observations, {saved} new rows saved")
        print(f"  Range: {df_factor['timestamp'].iloc[0].date()} ~ {df_factor['timestamp'].iloc[-1].date()}")
        return saved

    except Exception as e:
        print(f"  [ERROR] SGE Au99.99 fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


def compute_sge_premium():
    """
    计算上海金溢价 (SGE Premium) 并存入 factor_data

    公式:
      SGE Premium (USD/oz) = (SGE_price * 31.1035) / USDCNY - International_Gold_Price

    其中:
      - SGE_price: Au99.99 收盘价 (CNY/克)
      - 31.1035: 克/盎司转换系数
      - USDCNY: 美元兑人民币汇率
      - International_Gold_Price: 国际现货金价 (USD/oz)

    该指标量化了东方市场对黄金的溢价程度，
    是 2023-2024 黄金脱锚期间的核心观测因子。
    """
    print("\n" + "=" * 60)
    print("  [3/3] Computing SGE Premium (Shanghai Gold Premium)")
    print("=" * 60)

    factor_name = 'sge_premium'

    conn = sqlite3.connect(DB_PATH)

    df_sge = pd.read_sql_query(
        "SELECT timestamp, factor_value FROM factor_data WHERE symbol='MACRO' AND factor_name='sge_au9999' ORDER BY timestamp",
        conn
    )
    df_usdcny = pd.read_sql_query(
        "SELECT timestamp, factor_value FROM factor_data WHERE symbol='MACRO' AND factor_name='dexchus' ORDER BY timestamp",
        conn
    )
    df_gold = pd.read_sql_query(
        "SELECT timestamp, close FROM market_data_gold WHERE symbol='GCUSD' ORDER BY timestamp",
        conn
    )

    conn.close()

    if df_sge.empty:
        print("  [ERROR] No SGE Au99.99 data! Run fetch_sge_au9999 first.")
        return 0
    if df_usdcny.empty:
        print("  [ERROR] No DEXCHUS data! Run fetch_fred_dexchus first.")
        return 0
    if df_gold.empty:
        print("  [ERROR] No gold price data!")
        return 0

    df_sge['timestamp'] = pd.to_datetime(df_sge['timestamp']).dt.normalize()
    df_usdcny['timestamp'] = pd.to_datetime(df_usdcny['timestamp']).dt.normalize()
    df_gold['timestamp'] = pd.to_datetime(df_gold['timestamp']).dt.normalize()

    df_sge = df_sge.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
    df_usdcny = df_usdcny.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
    df_gold = df_gold.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()

    df_sge.rename(columns={'factor_value': 'sge_cny_per_gram'}, inplace=True)
    df_usdcny.rename(columns={'factor_value': 'usdcny'}, inplace=True)
    df_gold.rename(columns={'close': 'intl_gold_usd_per_oz'}, inplace=True)

    df = df_sge.join(df_usdcny, how='inner').join(df_gold, how='inner')
    print(f"  Three-series aligned: {len(df)} rows")

    if df.empty:
        print("  [ERROR] No overlapping data between SGE, USDCNY, and Gold!")
        return 0

    df['sge_usd_per_oz'] = (df['sge_cny_per_gram'] * GRAMS_PER_TROY_OZ) / df['usdcny']
    df['sge_premium'] = df['sge_usd_per_oz'] - df['intl_gold_usd_per_oz']

    latest_ts = _get_latest_factor_timestamp(factor_name)
    if latest_ts:
        latest_date = pd.Timestamp(latest_ts).normalize()
        df = df[df.index > latest_date]
        print(f"  Incremental from: {latest_date.date()}")

    if df.empty:
        print("  No new SGE Premium data after incremental filter!")
        return 0

    rows = []
    for ts, row in df.iterrows():
        rows.append({
            'timestamp': ts,
            'factor_name': factor_name,
            'factor_value': float(row['sge_premium']),
        })

    df_factor = pd.DataFrame(rows)
    df_factor = df_factor.drop_duplicates(subset=['timestamp', 'factor_name']).sort_values('timestamp')
    saved = _save_factor_data(df_factor)

    _save_factor_metadata(
        factor_name,
        'SGE Premium: Shanghai Gold vs International Gold (USD/oz) = (SGE * 31.1035 / USDCNY) - IntlGold',
        'Derived',
        'USD/oz',
        'D'
    )

    premium = df['sge_premium']
    print(f"  SGE Premium: {len(rows)} observations, {saved} new rows saved")
    print(f"  Range: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"  Stats: mean={premium.mean():.2f}, std={premium.std():.2f}, "
          f"min={premium.min():.2f}, max={premium.max():.2f}")
    print(f"  Recent 5 days:")
    for ts, row in df.tail(5).iterrows():
        print(f"    {ts.date()}: SGE={row['sge_cny_per_gram']:.2f} CNY/g, "
              f"USDCNY={row['usdcny']:.4f}, "
              f"IntlGold={row['intl_gold_usd_per_oz']:.2f}, "
              f"Premium={row['sge_premium']:+.2f} USD/oz")

    return saved


def verify_data():
    """验证所有 V8 所需数据的完整性"""
    print("\n" + "=" * 60)
    print("  Data Verification for V8")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    required_factors = {
        'dfii10': ('FRED', '10Y TIPS Real Rate'),
        'dtwexbgs': ('FRED', 'Trade-Weighted USD Index'),
        'bamlh0a0hym2': ('FRED', 'HY Credit Spread'),
        'walcl': ('FRED', 'Fed Total Assets'),
        'm2sl': ('FRED', 'M2 Money Supply'),
        'vixcls': ('FRED', 'VIX Index'),
        'dexchus': ('FRED', 'USD/CNY Exchange Rate'),
        'sge_au9999': ('AkShare', 'SGE Au99.99 Price'),
        'sge_premium': ('Derived', 'SGE Premium'),
    }

    all_ok = True
    for factor_name, (source, desc) in required_factors.items():
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM factor_data WHERE symbol='MACRO' AND factor_name=?",
            (factor_name,)
        )
        row = cursor.fetchone()
        if row and row[0] > 0:
            print(f"  ✅ {factor_name:<20s} [{source:<8s}] {row[0]:>6d} rows  {row[1][:10]} ~ {row[2][:10]}")
        else:
            print(f"  ❌ {factor_name:<20s} [{source:<8s}] MISSING!")
            all_ok = False

    conn.close()

    if all_ok:
        print("\n  All V8 data sources verified! ✅")
    else:
        print("\n  Some data sources missing! ❌")

    return all_ok


def main():
    log_path = os.path.join(os.path.dirname(__file__), 'fetch_v8_data_log.txt')
    log_file = open(log_path, 'w', encoding='utf-8')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log("=" * 78)
    log("  V8 Data Engineering - Macro Data Pool Expansion")
    log(f"  Database: {DB_PATH}")
    log("=" * 78)

    saved_dexchus = fetch_fred_dexchus(start_date='2007-01-01')
    saved_sge = fetch_sge_au9999(start_date='2007-01-01')
    saved_premium = compute_sge_premium()

    log("\n" + "=" * 78)
    log("  Summary")
    log("=" * 78)
    log(f"  DEXCHUS:     {saved_dexchus} new rows")
    log(f"  SGE Au99.99: {saved_sge} new rows")
    log(f"  SGE Premium: {saved_premium} new rows")

    verify_data()

    log("\n" + "=" * 78)
    log("  V8 Data Engineering Complete!")
    log("=" * 78)

    log_file.close()


if __name__ == '__main__':
    main()
