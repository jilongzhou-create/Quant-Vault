#!/usr/bin/env python3
"""
因子数据清理工具 - 清空 raw_data 和 factor_data 表
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

logger = get_logger("factor_data_cleaner")

def main():
    print("=" * 80)
    print("因子数据清理工具")
    print("=" * 80)
    
    print(f"数据库文件: {DB_PATH}")
    print()
    
    # 检查是否有自动确认参数
    auto_confirm = len(sys.argv) > 1 and sys.argv[1] == "--yes"
    
    # 确认操作
    if auto_confirm:
        confirm = 'y'
        print("自动确认执行操作")
    else:
        confirm = input("警告：此操作将清空 raw_data 和 factor_data 表中的所有数据！\n是否确认执行？(y/N): "
                       ).strip().lower()
    
    if confirm != 'y':
        print("操作已取消")
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 开始事务
        conn.execute('BEGIN TRANSACTION')
        
        # 清空 raw_data 表
        cursor.execute('DELETE FROM raw_data')
        raw_rows_deleted = cursor.rowcount
        print(f"已清空 raw_data 表，删除了 {raw_rows_deleted} 条记录")
        
        # 清空 factor_data 表
        cursor.execute('DELETE FROM factor_data')
        factor_rows_deleted = cursor.rowcount
        print(f"已清空 factor_data 表，删除了 {factor_rows_deleted} 条记录")
        
        # 提交事务
        conn.commit()
        print("事务提交成功")
        
        conn.close()
        
        print("\n" + "=" * 80)
        print("清理完成！")
        print(f"- raw_data: 删除了 {raw_rows_deleted} 条记录")
        print(f"- factor_data: 删除了 {factor_rows_deleted} 条记录")
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
