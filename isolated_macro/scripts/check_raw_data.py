import sqlite3, os, sys, json
sys.path.insert(0, r'c:\Users\Jilong\Documents\trae_projects\trading_agent')
from config import DB_PATH

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT id, source_id, event_timestamp, raw_content FROM raw_data WHERE source_id = 'fred_BAMLH0A0HYM2' ORDER BY event_timestamp ASC LIMIT 5")
rows = cur.fetchall()
print("raw_data BAMLH0A0HYM2 sample (earliest):")
for r in rows:
    print(f"  id={r[0]}, source={r[1]}, ts={r[2]}")
    try:
        content = json.loads(r[3]) if isinstance(r[3], str) else r[3]
        print(f"    raw_content keys: {list(content.keys()) if isinstance(content, dict) else type(content)}")
        if isinstance(content, dict):
            print(f"    date={content.get('date')}, value={content.get('value')}")
    except:
        print(f"    raw_content: {str(r[3])[:200]}")

cur.execute("SELECT id, source_id, event_timestamp, raw_content FROM raw_data WHERE source_id = 'fred_BAMLH0A0HYM2' AND event_timestamp LIKE '2008%' ORDER BY event_timestamp ASC LIMIT 5")
rows = cur.fetchall()
print(f"\nraw_data BAMLH0A0HYM2 2008 data: {len(rows)} rows")

cur.execute("SELECT COUNT(*) FROM raw_data WHERE source_id = 'fred_BAMLH0A0HYM2' AND event_timestamp < '2018-01-01'")
row = cur.fetchone()
print(f"raw_data before 2018: {row[0]} rows")

cur.execute("SELECT COUNT(*) FROM factor_data WHERE factor_name = 'bamlh0a0hym2' AND timestamp < '2018-01-01'")
row = cur.fetchone()
print(f"factor_data before 2018: {row[0]} rows")

cur.execute("SELECT timestamp, factor_value FROM factor_data WHERE factor_name = 'bamlh0a0hym2' ORDER BY timestamp ASC LIMIT 10")
rows = cur.fetchall()
print("\nfactor_data bamlh0a0hym2 earliest 10 rows:")
for r in rows:
    print(f"  {r[0]}: {r[1]}")

cur.execute("SELECT timestamp, factor_value FROM factor_data WHERE factor_name = 'bamlh0a0hym2' AND timestamp >= '2020-03-01' AND timestamp <= '2020-04-30' ORDER BY timestamp LIMIT 10")
rows = cur.fetchall()
print("\nfactor_data bamlh0a0hym2 2020 COVID crash:")
for r in rows:
    print(f"  {r[0]}: {r[1]}")

conn.close()
