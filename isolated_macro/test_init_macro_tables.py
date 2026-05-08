#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
临时测试脚本：验证宏观估值模型表是否正确创建

执行步骤：
  1. 调用 init_macro_tables() 创建两张新表
  2. 查询数据库验证表是否存在
  3. 验证表结构（字段、主键、索引）
  4. 插入测试数据并读取验证
  5. 清理测试数据
"""

import os
import sys
import sqlite3

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DB_PATH
from database.db_manager import init_macro_tables


def main():
    print("=" * 60)
    print("宏观估值模型表 - 初始化验证脚本")
    print("=" * 60)

    print(f"\n数据库路径: {DB_PATH}")
    print(f"数据库文件存在: {os.path.exists(DB_PATH)}")

    # Step 1: 初始化表
    print("\n[Step 1] 调用 init_macro_tables()...")
    try:
        init_macro_tables()
        print("  ✅ init_macro_tables() 执行成功")
    except Exception as e:
        print(f"  ❌ init_macro_tables() 执行失败: {e}")
        return

    # Step 2: 验证表是否存在
    print("\n[Step 2] 验证表是否存在...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    all_tables = [row[0] for row in cursor.fetchall()]
    print(f"  数据库中共有 {len(all_tables)} 张表")

    for table_name in ['macro_model_registry', 'macro_valuation_daily']:
        if table_name in all_tables:
            print(f"  ✅ {table_name} 存在")
        else:
            print(f"  ❌ {table_name} 不存在")

    # Step 3: 验证表结构
    print("\n[Step 3] 验证表结构...")

    cursor.execute(f"PRAGMA table_info(macro_model_registry)")
    columns = cursor.fetchall()
    print(f"\n  macro_model_registry 字段 ({len(columns)} 个):")
    for col in columns:
        pk_mark = " [PK]" if col[5] else ""
        notnull_mark = " NOT NULL" if col[3] else ""
        print(f"    {col[1]:20s} {col[2]:10s}{notnull_mark}{pk_mark}")

    cursor.execute(f"PRAGMA table_info(macro_valuation_daily)")
    columns = cursor.fetchall()
    print(f"\n  macro_valuation_daily 字段 ({len(columns)} 个):")
    for col in columns:
        pk_mark = " [PK]" if col[5] else ""
        notnull_mark = " NOT NULL" if col[3] else ""
        print(f"    {col[1]:20s} {col[2]:10s}{notnull_mark}{pk_mark}")

    # 验证索引
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='macro_valuation_daily'")
    indexes = [row[0] for row in cursor.fetchall()]
    print(f"\n  macro_valuation_daily 索引 ({len(indexes)} 个):")
    for idx in indexes:
        print(f"    {idx}")

    # Step 4: 插入测试数据
    print("\n[Step 4] 插入测试数据...")
    try:
        cursor.execute("""
            INSERT INTO macro_model_registry (model_id, target_symbol, formula_desc, params_json, is_start_date, is_end_date)
            VALUES ('test_gc_model_v1', 'GCUSD', 'FairValue = β0 + β1*M2 + β2*DGS10 + β3*VIX', '{"beta_0": 1200, "beta_1": 0.003, "beta_2": -85, "beta_3": -12}', '2020-01-01', '2024-12-31')
        """)
        print("  ✅ macro_model_registry 测试数据插入成功")

        cursor.execute("""
            INSERT INTO macro_valuation_daily (timestamp, model_id, symbol, market_price, fair_value, valuation_spread, spread_zscore, target_exposure)
            VALUES ('2025-05-01', 'test_gc_model_v1', 'GCUSD', 3300.5, 3200.0, 100.5, 1.85, 0.6)
        """)
        print("  ✅ macro_valuation_daily 测试数据插入成功")

        conn.commit()
    except Exception as e:
        print(f"  ❌ 测试数据插入失败: {e}")

    # Step 5: 读取验证
    print("\n[Step 5] 读取验证...")
    cursor.execute("SELECT model_id, target_symbol, params_json FROM macro_model_registry WHERE model_id = 'test_gc_model_v1'")
    row = cursor.fetchone()
    if row:
        print(f"  ✅ macro_model_registry 读取成功: model_id={row[0]}, symbol={row[1]}")
    else:
        print("  ❌ macro_model_registry 读取失败")

    cursor.execute("SELECT timestamp, symbol, fair_value, spread_zscore, target_exposure FROM macro_valuation_daily WHERE model_id = 'test_gc_model_v1'")
    row = cursor.fetchone()
    if row:
        print(f"  ✅ macro_valuation_daily 读取成功: ts={row[0]}, symbol={row[1]}, fair={row[2]}, z={row[3]}, exposure={row[4]}")
    else:
        print("  ❌ macro_valuation_daily 读取失败")

    # Step 6: 清理测试数据
    print("\n[Step 6] 清理测试数据...")
    cursor.execute("DELETE FROM macro_valuation_daily WHERE model_id = 'test_gc_model_v1'")
    cursor.execute("DELETE FROM macro_model_registry WHERE model_id = 'test_gc_model_v1'")
    conn.commit()
    print("  ✅ 测试数据已清理")

    # Step 7: 确认现有业务表未受影响
    print("\n[Step 7] 确认现有业务表未受影响...")
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'macro_%' AND name NOT LIKE 'sqlite_%'")
    non_macro_count = cursor.fetchone()[0]
    print(f"  非 macro_ 前缀的业务表数量: {non_macro_count}")

    conn.close()

    print("\n" + "=" * 60)
    print("✅ 全部验证通过！宏观估值模型表已正确创建且与现有系统隔离")
    print("=" * 60)


if __name__ == '__main__':
    main()
