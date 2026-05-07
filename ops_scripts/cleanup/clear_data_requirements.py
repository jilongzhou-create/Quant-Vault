#!/usr/bin/env python3
"""
清空 data_requirements 和 research_articles 表数据（保留表结构）
用于清空测试数据
"""

import sys
import os
import sqlite3

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:

    sys.path.insert(0, project_root)

from config import DB_PATH


def main():
    print("=" * 80)
    print("⚠️  清空 data_requirements 和 research_articles 表数据（保留表结构）")
    print("=" * 80)
    
    if not os.path.exists(DB_PATH):
        print(f"错误：数据库文件不存在: {DB_PATH}")
        return
    
    print(f"\n数据库文件: {DB_PATH}")
    print()
    
    # 确认
    confirm = input("⚠️  这将删除 data_requirements 和 research_articles 表里的所有数据！\n   确定要继续吗？(输入 YES 确认): ")
    
    if confirm.strip().upper() != "YES":
        print("操作已取消")
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 先查一下有多少条记录
        print("\n正在查询现有数据...")
        
        cursor.execute("SELECT COUNT(*) FROM data_requirements")
        req_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM research_articles")
        articles_count = cursor.fetchone()[0]
        
        print(f"  data_requirements:   {req_count} 条")
        print(f"  research_articles: {articles_count} 条")
        
        # 清空数据
        print("\n正在清空表...")
        
        cursor.execute("DELETE FROM data_requirements")
        deleted_req = cursor.rowcount
        
        cursor.execute("DELETE FROM research_articles")
        deleted_articles = cursor.rowcount
        
        conn.commit()
        
        print(f"\n✅ 清空完成！")
        print(f"  已删除 data_requirements:   {deleted_req} 条")
        print(f"  已删除 research_articles: {deleted_articles} 条")
        
        # 重置自增ID
        print("\n正在重置自增ID...")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='data_requirements'")
        conn.commit()
        print("✅ 自增ID已重置 (data_requirements)")
        
        # 验证
        print("\n验证结果:")
        cursor.execute("SELECT COUNT(*) FROM data_requirements")
        print(f"  data_requirements:   {cursor.fetchone()[0]} 条")
        
        cursor.execute("SELECT COUNT(*) FROM research_articles")
        print(f"  research_articles: {cursor.fetchone()[0]} 条")
        
        conn.close()
        
        print("\n" + "=" * 80)
        print("✅ 所有操作完成！data_requirements 和 research_articles 表已清空")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
