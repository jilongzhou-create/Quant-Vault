import sys, os, time, json, sqlite3
import numpy as np
import pandas as pd
sys.path.insert(0, r'c:\Users\Jilong\Documents\trae_projects\trading_agent')
from dotenv import load_dotenv
load_dotenv()
from config import DB_PATH

FRED_API_KEY = os.environ.get("FRED_API_KEY")
SERIES_ID = "BAA10YM"

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=3, backoff_factor=1)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

proxy = os.environ.get('HTTPS_PROXY', os.environ.get('HTTP_PROXY', ''))
proxies = {'http': proxy, 'https': proxy} if proxy else None

url = "https://api.stlouisfed.org/fred/series/observations"
params = {
    "series_id": SERIES_ID,
    "api_key": FRED_API_KEY,
    "file_type": "json",
    "observation_start": "1996-01-01",
    "observation_end": "2026-12-31",
    "sort_order": "asc",
}

print(f"Fetching {SERIES_ID} from FRED...")
resp = session.get(url, params=params, timeout=30, proxies=proxies)
resp.raise_for_status()

data = resp.json()
observations = data.get('observations', [])
print(f"  Got {len(observations)} observations")

if not observations:
    print("  No data returned!")
    sys.exit(1)

df = pd.DataFrame(observations)
df['date_parsed'] = pd.to_datetime(df['date'])
df = df[df['value'] != '.']

if df.empty:
    print("  All values are '.' (missing)")
    sys.exit(1)

print(f"  Valid observations: {len(df)}")
print(f"  Date range: {df['date_parsed'].min()} ~ {df['date_parsed'].max()}")
print(f"  Value range: {df['value'].astype(float).min():.2f} ~ {df['value'].astype(float).max():.2f}")

source_id = f"fred_{SERIES_ID}"
lag_days = 1

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

raw_count = 0
factor_count = 0

for _, row in df.iterrows():
    event_date = pd.Timestamp(row['date'])
    event_ts = event_date + pd.Timedelta(days=lag_days)
    event_ts_str = event_ts.strftime('%Y-%m-%d 23:59:59')
    value = float(row['value'])

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

    factor_ts = event_ts.strftime('%Y-%m-%dT23:59:59')
    cur.execute("""
        SELECT id FROM factor_data
        WHERE symbol = 'MACRO' AND factor_name = ? AND timestamp = ?
    """, (SERIES_ID.lower(), factor_ts))
    if cur.fetchone() is None:
        cur.execute("""
            INSERT INTO factor_data (symbol, timestamp, factor_name, factor_value)
            VALUES ('MACRO', ?, ?, ?)
        """, (factor_ts, SERIES_ID.lower(), value))
        factor_count += 1

conn.commit()
conn.close()

print(f"\n  Stored: raw_data={raw_count} new, factor_data={factor_count} new")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("SELECT COUNT(*), MIN(event_timestamp), MAX(event_timestamp) FROM raw_data WHERE source_id = ?", (source_id,))
row = cur.fetchone()
print(f"  raw_data total: {row[0]}, range: {row[1]} ~ {row[2]}")
cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM factor_data WHERE factor_name = ?", (SERIES_ID.lower(),))
row = cur.fetchone()
print(f"  factor_data total: {row[0]}, range: {row[1]} ~ {row[2]}")

cur.execute("SELECT timestamp, factor_value FROM factor_data WHERE factor_name = ? AND timestamp >= '2008-06-01' AND timestamp <= '2009-06-01' ORDER BY timestamp", (SERIES_ID.lower(),))
rows = cur.fetchall()
print(f"\n  2008 GFC period data:")
for r in rows:
    print(f"    {r[0]}: {r[1]}")

conn.close()
print("\nDone!")
