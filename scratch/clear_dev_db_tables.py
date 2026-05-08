import sqlite3
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())
from config import DB_PATH

def clear_dev_db():
    print(f"Targeting database: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print("Database file does not exist.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    
    print(f"Total tables found: {len(tables)}")
    
    for table in tables:
        if table == 'market_data_crypto':
            print(f"Skipping {table} (Protected)")
            continue
        if table.startswith('sqlite_'):
            print(f"Skipping {table} (Internal)")
            continue
            
        print(f"Dropping table: {table}...")
        cursor.execute(f'DROP TABLE "{table}"')
        
    conn.commit()
    conn.close()
    print("\n✅ All tables except 'market_data_crypto' have been dropped.")

if __name__ == "__main__":
    clear_dev_db()
