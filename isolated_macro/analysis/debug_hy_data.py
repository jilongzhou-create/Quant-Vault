import sys, os
sys.path.insert(0, '.')
from config import DB_PATH
import sqlite3

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("PRAGMA table_info(raw_data)")
cols = cur.fetchall()
print("raw_data columns:", [c[1] for c in cols])

cur.execute("PRAGMA table_info(factor_data)")
cols = cur.fetchall()
print("factor_data columns:", [c[1] for c in cols])

cur.execute("SELECT * FROM raw_data WHERE source_id = 'fred_BAMLH0A0HYM2' LIMIT 3")
rows = cur.fetchall()
col_names = [desc[0] for desc in cur.description]
print(f"\nraw_data sample (cols={col_names}):")
for r in rows:
    print(f"  {r}")

cur.execute("SELECT * FROM factor_data WHERE factor_name = 'bamlh0a0hym2' LIMIT 3")
rows = cur.fetchall()
col_names = [desc[0] for desc in cur.description]
print(f"\nfactor_data sample (cols={col_names}):")
for r in rows:
    print(f"  {r}")

cur.execute("SELECT COUNT(*), MIN(event_timestamp), MAX(event_timestamp) FROM raw_data WHERE source_id = 'fred_BAMLH0A0HYM2'")
row = cur.fetchone()
print(f"\nraw_data BAMLH0A0HYM2: count={row[0]}, min_ts={row[1]}, max_ts={row[2]}")

cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM factor_data WHERE factor_name = 'bamlh0a0hym2'")
row = cur.fetchone()
print(f"factor_data bamlh0a0hym2: count={row[0]}, min_ts={row[1]}, max_ts={row[2]}")

conn.close()
