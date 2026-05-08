import sqlite3, os, sys
sys.path.insert(0, r'c:\Users\Jilong\Documents\trae_projects\trading_agent')
from config import DB_PATH
print('DB_PATH:', DB_PATH)
print('DB exists:', os.path.exists(DB_PATH))

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT COUNT(*), MIN(event_timestamp), MAX(event_timestamp) FROM raw_data WHERE source_id = 'fred_BAMLH0A0HYM2'")
row = cur.fetchone()
print(f'raw_data BAMLH0A0HYM2: count={row[0]}, range={row[1]} ~ {row[2]}')

cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM factor_data WHERE factor_name = 'bamlh0a0hym2'")
row = cur.fetchone()
print(f'factor_data bamlh0a0hym2: count={row[0]}, range={row[1]} ~ {row[2]}')

cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM factor_data WHERE factor_name = 'bamlh0a0hym2' AND timestamp >= '2007-01-01' AND timestamp <= '2009-12-31'")
row = cur.fetchone()
print(f'factor_data 2007-2009: count={row[0]}, range={row[1]} ~ {row[2]}')

cur.execute("SELECT timestamp, factor_value FROM factor_data WHERE factor_name = 'bamlh0a0hym2' AND timestamp >= '2008-09-01' AND timestamp <= '2008-12-31' ORDER BY timestamp LIMIT 10")
rows = cur.fetchall()
print(f'2008 crisis data sample ({len(rows)} rows):')
for r in rows:
    print(f'  {r[0]}: {r[1]}')

cur.execute("SELECT timestamp, factor_value FROM factor_data WHERE factor_name = 'bamlh0a0hym2' ORDER BY timestamp ASC LIMIT 5")
rows = cur.fetchall()
print(f'Earliest data ({len(rows)} rows):')
for r in rows:
    print(f'  {r[0]}: {r[1]}')

cur.execute("SELECT COUNT(*) FROM factor_data WHERE factor_name = 'sge_premium'")
row = cur.fetchone()
print(f'sge_premium count: {row[0]}')

conn.close()
