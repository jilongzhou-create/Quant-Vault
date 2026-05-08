#!/usr/bin/env python3
"""
补拉 BAMLH0A0HYM2 (高收益债利差) 1996-2017 历史数据

问题: raw_data 中该序列仅从 2018 年开始, 缺失 2007-2017 的关键数据
      (2008 GFC 期间利差从 3% 飙升至 20%+, 是信用恐慌因子的核心输入)
方案: 直接调用 FRED API 补拉, 仅存入 raw_data (原始水平值)
      CreditPanicFactor v2 从 raw_data 读取原始利差, 自行计算 252 日 Z-Score
注意: 不写入 factor_data, 避免与已有的 Z-Score 标准化值冲突
"""

import sys
import os
import time
import json
import sqlite3
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DB_PATH
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.environ.get("FRED_API_KEY")
SERIES_ID = "BAMLH0A0HYM2"
FACTOR_NAME = "bamlh0a0hym2"


def fetch_and_store():
    if not FRED_API_KEY:
        print("[ERROR] FRED_API_KEY not configured!")
        return

    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": SERIES_ID,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": "1996-12-31",
        "observation_end": "2017-12-31",
    }

    print(f"Fetching {SERIES_ID} from 1996-12-31 to 2017-12-31...")
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(0.6)

    data = resp.json()
    observations = data.get('observations', [])
    print(f"  Got {len(observations)} observations")

    if not observations:
        print("  No data returned!")
        return

    df = pd.DataFrame(observations)
    df['date_parsed'] = pd.to_datetime(df['date'])
    df = df[df['value'] != '.']

    if df.empty:
        print("  All values are '.' (missing)")
        return

    print(f"  Valid observations: {len(df)}")
    print(f"  Date range: {df['date_parsed'].min()} ~ {df['date_parsed'].max()}")
    print(f"  Value range: {df['value'].astype(float).min():.2f} ~ {df['value'].astype(float).max():.2f}")

    lag_days = 1
    source_id = f"fred_{SERIES_ID}"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    raw_count = 0

    for _, row in df.iterrows():
        event_date = pd.Timestamp(row['date'])
        event_ts = event_date + pd.Timedelta(days=lag_days)
        event_ts_str = event_ts.strftime('%Y-%m-%d 23:59:59')

        raw_content = json.dumps({
            "realtime_start": row.get('realtime_start', ''),
            "realtime_end": row.get('realtime_end', ''),
            "date": row['date'],
            "value": row['value'],
            "date_parsed": row['date_parsed'].isoformat(),
            "event_timestamp": event_ts_str,
            "source_id": source_id,
        })

        cur.execute("""
            SELECT id FROM raw_data
            WHERE source_id = ? AND event_timestamp = ?
        """, (source_id, event_ts_str))
        if cur.fetchone() is None:
            cur.execute("""
                INSERT INTO raw_data (source_id, event_timestamp, fetch_timestamp, raw_content)
                VALUES (?, ?, datetime('now'), ?)
            """, (source_id, event_ts_str, raw_content))
            raw_count += 1

    conn.commit()
    conn.close()

    print(f"\n  Stored: raw_data={raw_count} new (factor_data NOT touched)")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), MIN(event_timestamp), MAX(event_timestamp) FROM raw_data WHERE source_id = ?", (source_id,))
    row = cur.fetchone()
    print(f"  raw_data total: {row[0]}, range: {row[1]} ~ {row[2]}")
    cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM factor_data WHERE factor_name = ?", (FACTOR_NAME,))
    row = cur.fetchone()
    print(f"  factor_data total: {row[0]}, range: {row[1]} ~ {row[2]}")
    conn.close()


if __name__ == '__main__':
    fetch_and_store()
