import os
import sys
import sqlite3

# Ensure root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from content_engine import data_extractor
from config import DB_PATH

def debug_pipeline_a():
    print(f"Using DB_PATH: {DB_PATH}")
    
    # 1. 直接用 data_extractor 查
    results = data_extractor.get_pipeline_a_strategies(limit=5)
    print(f"data_extractor.get_pipeline_a_strategies() found: {len(results)} strategies")
    for r in results:
        print(f" - {r['name']} (Annualized Return: {r.get('metric_annualized_return')}, Status: {r.get('publish_status', 'UNKNOWN')})")
        
    # 2. 数据库原始查询
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 查一下所有 RESEARCHER_AGENT 且 annualized_return < 0 的策略，不管状态
    cursor.execute("""
        SELECT sd.name, sv.metric_annualized_return, sd.publish_status 
        FROM strategy_directions sd 
        JOIN strategy_versions sv ON sd.best_version_id = sv.ver_id 
        WHERE sd.source = 'RESEARCHER_AGENT' AND sv.metric_annualized_return < 0
    """)
    all_candidates = cursor.fetchall()
    print(f"\nTotal candidates in DB (regardless of status): {len(all_candidates)}")
    for name, ret, status in all_candidates:
        print(f" - {name}: Return={ret}, Status={status}")
        
    # 3. 查一下是否有 best_version_id 为 NULL 的情况
    cursor.execute("SELECT COUNT(*) FROM strategy_directions WHERE best_version_id IS NULL")
    null_count = cursor.fetchone()[0]
    print(f"\nStrategies with NULL best_version_id: {null_count}")
    
    conn.close()

if __name__ == "__main__":
    debug_pipeline_a()
