#!/usr/bin/env python3
import os
import sys
import sqlite3

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DB_PATH

def reset_all_publish_status():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM strategy_directions WHERE publish_status != 'UNPUBLISHED'")
    count = cursor.fetchone()[0]
    print(f"Found {count} strategies with non-UNPUBLISHED status. Resetting...")
    cursor.execute("UPDATE strategy_directions SET publish_status = 'UNPUBLISHED'")
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM strategy_directions WHERE publish_status = 'UNPUBLISHED'")
    total = cursor.fetchone()[0]
    conn.close()
    print(f"Done. {total} strategies now set to UNPUBLISHED.")

if __name__ == "__main__":
    reset_all_publish_status()
