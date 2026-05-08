import sqlite3
import os
import sys
sys.path.append(os.getcwd())
from config import DB_PATH
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(f"Remaining tables: {[t[0] for t in cursor.fetchall()]}")
conn.close()
