#!/usr/bin/env python3
"""
临时脚本 - 重置所有 data_requirements 表的状态为 PENDING
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:

    sys.path.insert(0, project_root)

from logger_setup import get_logger
from database.db_manager import get_pending_data_requirements

logger = get_logger("reset_data_requirements")


def reset_all_status():
    """
    重置所有数据需求状态为 PENDING
    """
    import sqlite3
    from database.db_manager import DB_PATH
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 先统计当前各状态的数量
        cursor.execute('SELECT status, COUNT(*) FROM data_requirements GROUP BY status')
        status_counts = cursor.fetchall()
        
        print("=" * 80)
        print("数据需求状态重置工具")
        print("=" * 80)
        print("\n当前各状态数量：")
        for status, count in status_counts:
            print(f"  {status}: {count}")
        
        cursor.execute('SELECT COUNT(*) FROM data_requirements')
        total_count = cursor.fetchone()[0]
        
        print(f"\n总计: {total_count} 条数据需求")
        
        confirm = input("\n⚠️  警告：即将重置所有数据需求状态为 PENDING！\n是否继续？(yes/NO): ").strip().lower()
        
        if confirm != 'yes':
            print("\n操作取消")
            return
        
        # 执行重置
        cursor.execute("UPDATE data_requirements SET status = 'PENDING'")
        conn.commit()
        
        updated_count = cursor.rowcount
        
        print(f"\n✅ 成功重置 {updated_count} 条数据需求状态为 PENDING！")
        
        # 验证结果
        cursor.execute('SELECT status, COUNT(*) FROM data_requirements GROUP BY status')
        new_status_counts = cursor.fetchall()
        
        print("\n重置后的状态：")
        for status, count in new_status_counts:
            print(f"  {status}: {count}")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        logger.error(f"重置数据需求状态失败: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()


if __name__ == "__main__":
    reset_all_status()
