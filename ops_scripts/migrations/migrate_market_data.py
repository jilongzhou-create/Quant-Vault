#!/usr/bin/env python3
"""
迁移脚本：将旧表 market_data 的数据迁移到新表 market_data_crypto
"""

import os
import sys
import sqlite3
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DB_PATH
from logger_setup import get_logger

logger = get_logger("migrate_market_data")

def migrate():
    print("=" * 80)
    print("Database Migration: market_data -> market_data_crypto")
    print(f"Database: {DB_PATH}")
    print("=" * 80)
    
    if not os.path.exists(DB_PATH):
        print(f"Error: Database does not exist: {DB_PATH}")
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. Check old table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data'")
        if not cursor.fetchone():
            print("Old table market_data does not exist, skipping.")
            return
            
        # 2. Check new table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data_crypto'")
        if not cursor.fetchone():
            print("New table market_data_crypto does not exist, initializing...")
            from database.db_manager import init_db
            init_db()
            
        # 3. Get counts
        cursor.execute("SELECT COUNT(*) FROM market_data")
        old_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM market_data_crypto")
        new_count = cursor.fetchone()[0]
        
        print(f"Status:")
        print(f"  - market_data:        {old_count} rows")
        print(f"  - market_data_crypto: {new_count} rows")
        
        if old_count == 0:
            print("market_data is empty, skipping.")
            return
            
        # 4. Migrate
        print(f"\nMigrating {old_count} rows...")
        start_time = time.time()
        
        # INSERT OR IGNORE
        cursor.execute('''
        INSERT OR IGNORE INTO market_data_crypto 
        (symbol, timestamp, open, high, low, close, volume, rsi_14, macd, macd_signal, macd_hist)
        SELECT symbol, timestamp, open, high, low, close, volume, rsi_14, macd, macd_signal, macd_hist 
        FROM market_data
        ''')
        
        inserted_count = cursor.rowcount
        conn.commit()
        
        end_time = time.time()
        print(f"Migration completed! Time: {end_time - start_time:.2f} s")
        print(f"  - Migrated/Updated: {inserted_count} rows")
        
        # 5. Final check
        cursor.execute("SELECT COUNT(*) FROM market_data_crypto")
        final_new_count = cursor.fetchone()[0]
        print(f"  - market_data_crypto final count: {final_new_count} rows")
        
        # 6. Drop old table
        confirm = input("\nDrop old table market_data? (YES/NO): ")
        if confirm.strip().upper() == "YES":
            cursor.execute("DROP TABLE market_data")
            conn.commit()
            print("Old table market_data dropped.")
        else:
            print("Keeping old table as backup.")
            
        conn.close()
        print("\n" + "=" * 80)
        print("✅ 迁移任务结束")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 迁移失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    migrate()
