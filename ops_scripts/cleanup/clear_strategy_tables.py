#!/usr/bin/env python3
"""
策略表清理工具 - 清空策略版本相关表
"""

import sys
import os
import sqlite3

# 添加项目根目录到系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DB_PATH
from logger_setup import get_logger

logger = get_logger("strategy_cleaner")

def main():
    print("=" * 80)
    print("策略表清理工具")
    print("=" * 80)
    
    print(f"数据库文件: {DB_PATH}")
    print()
    
    # 确认操作
    confirm = input("警告：此操作将清空策略版本相关表的数据！\n是否确认执行？(y/N): "
                   ).strip().lower()
    
    if confirm != 'y':
        print("操作已取消")
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 开始事务
        conn.execute('BEGIN TRANSACTION')
        
        # 1. 清空 strategy_directions 表的 best_version_id 字段，设置 is_active_ensemble 为 0
        cursor.execute('UPDATE strategy_directions SET best_version_id = NULL, is_active_ensemble = 0')
        update_count = cursor.rowcount
        print(f"已更新 strategy_directions 表，影响 {update_count} 条记录")
        
        # 2. 清空 strategy_versions 表
        cursor.execute('DELETE FROM strategy_versions')
        delete_count = cursor.rowcount
        print(f"已清空 strategy_versions 表，删除了 {delete_count} 条记录")
        
        # 提交事务
        conn.commit()
        print("事务提交成功")
        
        conn.close()
        
        print("\n" + "=" * 80)
        print("清理完成！")
        print(f"- strategy_directions: 更新了 {update_count} 条记录")
        print(f"- strategy_versions: 删除了 {delete_count} 条记录")
        print("=" * 80)
        
    except Exception as e:
        print(f"清理失败: {e}")
        import traceback
        traceback.print_exc()
        # 回滚事务
        if 'conn' in locals():
            conn.rollback()
            conn.close()

if __name__ == "__main__":
    main()
