#!/usr/bin/env python3
import sys
import os

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sqlite3
from us_sector_ai_fund.config import DB_PATH, SECTOR_ETF_SYMBOLS, SECTOR_ETF_NAMES

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("=" * 70)
print("  market_data_us_sectors 数据验证")
print("=" * 70)

cursor.execute("SELECT COUNT(*) FROM market_data_us_sectors")
total = cursor.fetchone()[0]
print(f"\n总记录数: {total}")

print(f"\n各ETF记录数:")
for symbol in SECTOR_ETF_SYMBOLS:
    name = SECTOR_ETF_NAMES.get(symbol, '')
    cursor.execute("SELECT COUNT(*) FROM market_data_us_sectors WHERE symbol = ?", (symbol,))
    count = cursor.fetchone()[0]
    cursor.execute("SELECT MIN(date), MAX(date) FROM market_data_us_sectors WHERE symbol = ?", (symbol,))
    min_date, max_date = cursor.fetchone()
    print(f"  {symbol:5s} ({name:25s}): {count:>5} 条  [{min_date} ~ {max_date}]")

cursor.execute("""
    SELECT date, COUNT(DISTINCT symbol) as n_symbols
    FROM market_data_us_sectors
    GROUP BY date
    ORDER BY date
    LIMIT 5
""")
print(f"\n最早5个交易日的ETF数量:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} 只ETF")

cursor.execute("""
    SELECT date, COUNT(DISTINCT symbol) as n_symbols
    FROM market_data_us_sectors
    GROUP BY date
    ORDER BY date DESC
    LIMIT 5
""")
print(f"\n最近5个交易日的ETF数量:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} 只ETF")

conn.close()
