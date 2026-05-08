#!/usr/bin/env python3
"""
TLT MOVE Index Fetcher - 抓取 ICE BofA US Move Index (债市VIX)
直接通过 FRED API 获取并落库到 raw_data + factor_data
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import requests
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
except ImportError:
    pass

FRED_API_KEY = os.environ.get('FRED_API_KEY', '')

from config import DB_PATH


def fetch_move_raw():
    source_id = 'fred_MOVE'
    series_id = 'MOVE'

    print(f"[MOVE] Fetching {series_id} from FRED API...")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': '2007-01-01',
        'sort_order': 'asc',
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    observations = data.get('observations', [])
    print(f"  Got {len(observations)} observations from FRED")

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()

    saved = 0
    for obs in observations:
        ts = obs['date']
        val = obs['value']
        if val == '.':
            continue

        raw_content = json.dumps({'value': float(val), 'date': ts, 'series_id': series_id})
        fetch_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO raw_data (source_id, event_timestamp, fetch_timestamp, raw_content)
                VALUES (?, ?, ?, ?)
            ''', (source_id, ts, fetch_ts, raw_content))
            saved += 1
        except Exception as e:
            print(f"  [WARN] Failed to insert {ts}: {e}")

    conn.commit()
    conn.close()
    print(f"  Saved {saved} rows to raw_data (source_id={source_id})")
    return saved


def process_move_factor():
    source_id = 'fred_MOVE'
    factor_name = 'move'

    print(f"[MOVE] Processing factor from raw_data...")

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()

    cursor.execute('''
        SELECT event_timestamp, raw_content FROM raw_data
        WHERE source_id = ?
        ORDER BY event_timestamp
    ''', (source_id,))

    rows = cursor.fetchall()
    if not rows:
        print("  [ERROR] No raw data found for MOVE")
        conn.close()
        return

    records = []
    for ts, raw_json in rows:
        try:
            data = json.loads(raw_json)
            val = float(data['value'])
            records.append({'timestamp': pd.to_datetime(ts), 'value': val})
        except:
            continue

    df = pd.DataFrame(records)
    if df.empty:
        print("  [ERROR] No valid records")
        conn.close()
        return

    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"  Processing {len(df)} records, {df['timestamp'].min().date()} ~ {df['timestamp'].max().date()}")

    df['zscore_252'] = (df['value'] - df['value'].rolling(252, min_periods=63).mean()) / (
        df['value'].rolling(252, min_periods=63).std() + 1e-6
    )
    df['zscore_252'] = df['zscore_252'].fillna(0.0)

    saved = 0
    for _, row in df.iterrows():
        ts = row['timestamp'].strftime('%Y-%m-%d')
        val = row['value']
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO factor_data (symbol, timestamp, factor_name, factor_value)
                VALUES ('MACRO', ?, ?, ?)
            ''', ('MACRO', ts, factor_name, val))
            saved += 1
        except Exception as e:
            pass

    conn.commit()
    conn.close()
    print(f"  Saved {saved} rows to factor_data (factor_name={factor_name})")


def main():
    if not FRED_API_KEY:
        print("[ERROR] FRED_API_KEY not set in environment variables!")
        print("  Please set FRED_API_KEY before running this script.")
        return
    fetch_move_raw()
    process_move_factor()
    print("\n[DONE] MOVE index data ready for TLT factor mining!")


if __name__ == '__main__':
    import io
    _old = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        main()
    finally:
        sys.stdout = _old
    output = buf.getvalue()
    with open(os.path.join(os.path.dirname(__file__), 'move_fetch_report.txt'), 'w') as f:
        f.write(output)
    print(output)
