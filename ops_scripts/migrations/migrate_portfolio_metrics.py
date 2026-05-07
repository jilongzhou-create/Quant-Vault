#!/usr/bin/env python3
"""
组合表字段扩容迁移脚本
为 portfolios 表添加回测业绩指标字段：
  - metric_annualized_return (REAL)
  - metric_sharpe (REAL)
  - metric_max_drawdown (REAL)
  - weight_mode (TEXT)

幂等设计：重复运行不会报错，已存在的字段会跳过。
同时修改 PROD 和 DEV 数据库。
"""

import sqlite3
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DATA_DIR
from logger_setup import get_logger

logger = get_logger(__name__)

DB_PROD = os.path.join(DATA_DIR, 'trading_system_prod.db')
DB_DEV = os.path.join(DATA_DIR, 'trading_system_dev.db')

FIELDS_TO_ADD = [
    ('metric_annualized_return', 'REAL'),
    ('metric_sharpe', 'REAL'),
    ('metric_max_drawdown', 'REAL'),
    ('weight_mode', 'TEXT'),
]


def migrate_database(db_path):
    if not os.path.exists(db_path):
        logger.warning(f"数据库不存在，跳过: {db_path}")
        return

    logger.info(f"========== 开始迁移: {db_path} ==========")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(portfolios)")
    existing_fields = {row[1] for row in cursor.fetchall()}

    for field_name, field_type in FIELDS_TO_ADD:
        if field_name in existing_fields:
            print(f"  ⏭️  字段已存在，跳过: {field_name} ({field_type})")
        else:
            alter_sql = f"ALTER TABLE portfolios ADD COLUMN {field_name} {field_type}"
            cursor.execute(alter_sql)
            print(f"  ✅ 已添加字段: {field_name} ({field_type})")

    conn.commit()
    conn.close()
    logger.info(f"========== 迁移完成: {db_path} ==========\n")


def main():
    print("=" * 60)
    print("  组合表字段扩容迁移脚本")
    print("  添加: metric_annualized_return, metric_sharpe,")
    print("        metric_max_drawdown, weight_mode")
    print("  同时修改 PROD 和 DEV 数据库")
    print("=" * 60)
    print()

    for db_path in [DB_PROD, DB_DEV]:
        try:
            migrate_database(db_path)
        except Exception as e:
            logger.error(f"迁移失败 {db_path}: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ 迁移全部完成！")


if __name__ == "__main__":
    main()
