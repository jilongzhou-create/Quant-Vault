#!/usr/bin/env python3
"""
查询数据库所有表和字段信息
用于手动运行，为更新系统文档做准备
"""

import sys
import os
import sqlite3
import json
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from config import DB_PATH

def main():
    print("=" * 80)
    print("数据库结构查询工具")
    print(f"查询时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if not os.path.exists(DB_PATH):
        print(f"错误：数据库文件不存在: {DB_PATH}")
        return
    
    print(f"数据库文件: {DB_PATH}")
    print(f"文件是否存在: {os.path.exists(DB_PATH)}")
    print()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = cursor.fetchall()
        
        print(f"数据库共有 {len(tables)} 张表:")
        print("-" * 80)
        
        schema_info = {}
        
        for table in tables:
            table_name = table[0]
            print(f"\n【表】: {table_name}")
            print("-" * 60)
            
            # 获取表结构
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            print(f"  字段信息:")
            print(f"    {'序号':<6} {'字段名':<25} {'类型':<15} {'是否主键':<10}")
            print(f"    {'-'*6} {'-'*25} {'-'*15} {'-'*10}")
            
            table_columns = []
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, pk = col
                is_pk = "是" if pk == 1 else "否"
                print(f"    {col_id:<6} {col_name:<25} {col_type:<15} {is_pk:<10}")
                
                table_columns.append({
                    "id": col_id,
                    "name": col_name,
                    "type": col_type,
                    "not_null": not_null == 1,
                    "default": default_val,
                    "primary_key": pk == 1
                })
            
            # 获取索引信息
            cursor.execute(f"PRAGMA index_list({table_name});")
            indexes = cursor.fetchall()
            
            if indexes:
                print(f"\n  索引信息:")
                table_indexes = []
                for idx in indexes:
                    # 通用处理，不管返回多少字段，取前几个有用的
                    idx_name = idx[0] if len(idx) > 0 else "unknown"
                    unique_flag = idx[1] if len(idx) > 1 else 0
                    is_unique = "唯一" if unique_flag == 1 else "普通"
                    print(f"    - {idx_name} ({is_unique})")
                    table_indexes.append({
                        "name": idx_name,
                        "unique": unique_flag == 1
                    })
            else:
                table_indexes = []
            
            # 获取行数
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]
                print(f"\n  数据行数: {row_count}")
            except:
                row_count = 0
            
            schema_info[table_name] = {
                "columns": table_columns,
                "indexes": table_indexes,
                "row_count": row_count
            }
        
        # 保存到 JSON 文件
        output_file = os.path.join(project_root, 'db_schema.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schema_info, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 80)
        print(f"数据库结构已保存到: {output_file}")
        print("=" * 80)
        
        conn.close()
        
    except Exception as e:
        print(f"查询失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
