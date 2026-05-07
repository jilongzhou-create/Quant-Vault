#!/usr/bin/env python3
"""
查看 data_requirements 表数据
"""

import sys
import os
import sqlite3
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:

    sys.path.insert(0, project_root)

from config import DB_PATH


def main():
    parser = argparse.ArgumentParser(description="查看 data_requirements 表数据")
    parser.add_argument("-n", "--num", type=int, default=10,
                        help="显示前 N 条记录（默认 10 条）")
    parser.add_argument("-a", "--all", action="store_true",
                        help="显示所有记录")
    args = parser.parse_args()
    
    print("=" * 120)
    print("📋 查看 data_requirements 表数据")
    print("=" * 120)
    
    if not os.path.exists(DB_PATH):
        print(f"错误：数据库文件不存在: {DB_PATH}")
        return
    
    print(f"\n数据库文件: {DB_PATH}")
    print()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 先查一下总共有多少条记录
        cursor.execute("SELECT COUNT(*) FROM data_requirements")
        total_count = cursor.fetchone()[0]
        
        print(f"总记录数: {total_count} 条")
        print()
        
        if total_count == 0:
            print("data_requirements 表是空的")
            conn.close()
            return
        
        # 确定要显示的记录数
        limit = None
        if args.all:
            print(f"显示所有 {total_count} 条记录")
        else:
            limit = args.num
            print(f"显示前 {min(limit, total_count)} 条记录")
        
        print()
        print("=" * 120)
        
        # 查询数据
        if limit:
            cursor.execute("SELECT * FROM data_requirements ORDER BY id DESC LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT * FROM data_requirements ORDER BY id DESC")
        
        rows = cursor.fetchall()
        
        # 获取列名
        cursor.execute("PRAGMA table_info(data_requirements)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # 打印每条记录
        for idx, row in enumerate(rows, 1):
            print(f"\n【记录 {idx}/{len(rows)}】")
            print("-" * 120)
            
            for col_idx, col_name in enumerate(columns):
                value = row[col_idx]
                if value is not None:
                    # 格式化长文本
                    if col_name in ['required_reason', 'pending_strategy_desc'] and len(str(value)) > 100:
                        print(f"  {col_name:25s}: {str(value)[:100]}...")
                    else:
                        print(f"  {col_name:25s}: {value}")
                else:
                    print(f"  {col_name:25s}: NULL")
        
        print()
        print("=" * 120)
        print(f"\n显示完成！共显示 {len(rows)} 条记录（总记录数: {total_count}）")
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
