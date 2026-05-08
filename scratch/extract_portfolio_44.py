import sqlite3
import json
import os

conn = sqlite3.connect('data/trading_system_prod.db')
cursor = conn.cursor()

cursor.execute('SELECT dir_id FROM portfolio_components WHERE portfolio_id = 44')
dir_ids = [row[0] for row in cursor.fetchall()]

os.makedirs('scratch/portfolio_44_strategies', exist_ok=True)

for i, dir_id in enumerate(dir_ids):
    cursor.execute('''
        SELECT sv.code_content, sv.params_json, sd.name
        FROM strategy_directions sd
        JOIN strategy_versions sv ON sd.best_version_id = sv.ver_id
        WHERE sd.dir_id = ?
    ''', (dir_id,))
    row = cursor.fetchone()
    if row:
        code, params, name = row
        safe_name = name.replace(' ', '_').replace('/', '_')
        with open(f'scratch/portfolio_44_strategies/{i+1}_{safe_name}.py', 'w', encoding='utf-8') as f:
            f.write(code if code else '# NO CODE')
        with open(f'scratch/portfolio_44_strategies/{i+1}_{safe_name}_params.json', 'w', encoding='utf-8') as f:
            json.dump(json.loads(params) if params else {}, f, indent=2, ensure_ascii=False)
        print(f'Saved strategy {i+1}: {name}')

conn.close()
print('\nAll strategies extracted.')
