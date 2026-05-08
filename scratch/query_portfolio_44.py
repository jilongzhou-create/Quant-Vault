import sqlite3
import json

conn = sqlite3.connect('data/trading_system_prod.db')
cursor = conn.cursor()

cursor.execute('SELECT dir_id FROM portfolio_components WHERE portfolio_id = 44')
dir_ids = [row[0] for row in cursor.fetchall()]
print(f'Portfolio 44 has {len(dir_ids)} strategies:')
print()

for dir_id in dir_ids:
    cursor.execute('SELECT dir_id, name, description, best_version_id, target_symbol FROM strategy_directions WHERE dir_id = ?', (dir_id,))
    row = cursor.fetchone()
    if row:
        desc = row[2][:200] if row[2] else 'N/A'
        print(f'  dir_id: {row[0]}')
        print(f'  name: {row[1]}')
        print(f'  description: {desc}')
        print(f'  best_version_id: {row[3]}')
        print(f'  target_symbol: {row[4]}')
        print()

conn.close()
