import sqlite3
conn = sqlite3.connect('data/trading_system_prod.db')
r = conn.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM market_data_crypto WHERE symbol='BTC_USDT'").fetchone()
with open('scratch/check_result.txt', 'w') as f:
    f.write(f"BTC: {r[0]} ~ {r[1]}, total={r[2]}\n")
conn.close()
