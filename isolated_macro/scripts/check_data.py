import sqlite3
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DB_PATH

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT DISTINCT factor_name FROM factor_data WHERE symbol='MACRO' ORDER BY factor_name")
factors = [r[0] for r in cursor.fetchall()]
print('Existing MACRO factors:')
for f in factors:
    print(f'  {f}')
print(f'\nTotal: {len(factors)}')

for f in ['bamlh0a0hym2', 'walcl', 'm2sl', 'vixcls', 'dfii10', 'dtwexbgs']:
    cursor.execute('SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM factor_data WHERE factor_name=?', (f,))
    row = cursor.fetchone()
    if row and row[0] > 0:
        print(f'\n{f}: {row[0]} rows, {row[1]} ~ {row[2]}')
    else:
        print(f'\n{f}: NOT IN factor_data')

cursor.execute("SELECT DISTINCT source_id FROM raw_data WHERE source_id LIKE 'fred_%' ORDER BY source_id")
sources = [r[0] for r in cursor.fetchall()]
print('\nFRED raw_data sources:')
for s in sources:
    cursor.execute('SELECT COUNT(*) FROM raw_data WHERE source_id=?', (s,))
    cnt = cursor.fetchone()[0]
    print(f'  {s}: {cnt} rows')

conn.close()
