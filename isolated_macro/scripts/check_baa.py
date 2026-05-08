import sqlite3, os, sys
sys.path.insert(0, r'c:\Users\Jilong\Documents\trae_projects\trading_agent')
from config import DB_PATH

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

for factor_name in ['baa10ym', 'aaa10ym', 'bamlh0a0hym2']:
    cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM factor_data WHERE factor_name = ?", (factor_name,))
    row = cur.fetchone()
    if row[0] > 0:
        print(f"factor_data {factor_name}: count={row[0]}, range={row[1]} ~ {row[2]}")
    else:
        print(f"factor_data {factor_name}: EMPTY")

for source_id in ['fred_BAA10YM', 'fred_AAA10YM', 'fred_BAMLH0A0HYM2']:
    cur.execute("SELECT COUNT(*), MIN(event_timestamp), MAX(event_timestamp) FROM raw_data WHERE source_id = ?", (source_id,))
    row = cur.fetchone()
    if row[0] > 0:
        print(f"raw_data {source_id}: count={row[0]}, range={row[1]} ~ {row[2]}")
    else:
        print(f"raw_data {source_id}: EMPTY")

conn.close()
