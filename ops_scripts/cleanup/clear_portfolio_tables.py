
#!/usr/bin/env python3
"""
清空 Portfolio 相关表的脚本
警告：此操作不可逆！会删除所有组合数据！
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:

    sys.path.insert(0, project_root)

from database.db_manager import DB_PATH
import sqlite3

def clear_portfolio_tables():
    """
    清空三张 portfolio 相关表：
    - portfolios
    - portfolio_components
    - portfolio_daily_records
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        print("=" * 80)
        print("⚠️  警告：此操作将删除所有组合数据！".center(80))
        print("=" * 80)
        
        confirm = input("\n请输入 'YES' 确认删除: ").strip()
        
        if confirm != 'YES':
            print("\n❌ 操作已取消")
            return
        
        print("\n⏳ 正在清空 portfolio 相关表...")
        
        # 清空三张表
        cursor.execute("DELETE FROM portfolio_daily_records")
        deleted_daily = cursor.rowcount
        
        cursor.execute("DELETE FROM portfolio_components")
        deleted_components = cursor.rowcount
        
        cursor.execute("DELETE FROM portfolios")
        deleted_portfolios = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print("\n✅ 清空完成！")
        print(f"   - 删除了 {deleted_portfolios} 条组合记录")
        print(f"   - 删除了 {deleted_components} 条组合策略关联记录")
        print(f"   - 删除了 {deleted_daily} 条组合对账单记录")
        
    except Exception as e:
        print(f"\n❌ 清空失败: {e}")
        import traceback
        traceback.print_exc()
        if 'conn' in locals():
            conn.rollback()
            conn.close()

if __name__ == "__main__":
    clear_portfolio_tables()

