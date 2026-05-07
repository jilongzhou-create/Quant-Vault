#!/usr/bin/env python3
"""
多资产数据库重构迁移脚本

执行以下表结构变更：
1. us_market_data -> market_data_us_stock（重命名）
2. commodities_market_data -> market_data_commodity（重命名）
3. us_raw_data 数据迁移到 raw_data（合并）
4. commodities_raw_data 数据迁移到 raw_data（合并）
5. us_factor_data 数据迁移到 factor_data（合并）
6. commodities_factor_data 数据迁移到 factor_data（合并）
7. 删除已废弃的旧表

注意：此脚本会同时修改 PROD 和 DEV 数据库！
运行前请确保已备份数据库文件！
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


def migrate_database(db_path):
    """对单个数据库执行迁移"""
    if not os.path.exists(db_path):
        logger.warning(f"数据库不存在，跳过: {db_path}")
        return

    logger.info(f"========== 开始迁移: {db_path} ==========")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cursor.fetchall()}
    logger.info(f"现有表: {existing_tables}")

    # --- 1. 重命名 us_market_data -> market_data_us_stock ---
    if 'us_market_data' in existing_tables:
        if 'market_data_us_stock' in existing_tables:
            logger.warning("market_data_us_stock 已存在，跳过重命名 us_market_data")
        else:
            cursor.execute("ALTER TABLE us_market_data RENAME TO market_data_us_stock")
            logger.info("✓ us_market_data -> market_data_us_stock")
    else:
        logger.info("us_market_data 不存在，跳过")

    # --- 2. 重命名 commodities_market_data -> market_data_commodity ---
    if 'commodities_market_data' in existing_tables:
        if 'market_data_commodity' in existing_tables:
            logger.warning("market_data_commodity 已存在，跳过重命名 commodities_market_data")
        else:
            cursor.execute("ALTER TABLE commodities_market_data RENAME TO market_data_commodity")
            logger.info("✓ commodities_market_data -> market_data_commodity")
    else:
        logger.info("commodities_market_data 不存在，跳过")

    # --- 3. 合并 us_raw_data -> raw_data ---
    if 'us_raw_data' in existing_tables:
        if 'raw_data' in existing_tables:
            cursor.execute("""
                INSERT OR IGNORE INTO raw_data (source_id, event_timestamp, fetch_timestamp, raw_content)
                SELECT
                    symbol || '_' || data_type,
                    event_timestamp,
                    CURRENT_TIMESTAMP,
                    data_json
                FROM us_raw_data
            """)
            migrated = cursor.rowcount
            logger.info(f"✓ us_raw_data -> raw_data: 迁移 {migrated} 条")
        else:
            logger.warning("raw_data 表不存在，无法合并 us_raw_data")

    # --- 4. 合并 commodities_raw_data -> raw_data ---
    if 'commodities_raw_data' in existing_tables:
        if 'raw_data' in existing_tables:
            cursor.execute("""
                INSERT OR IGNORE INTO raw_data (source_id, event_timestamp, fetch_timestamp, raw_content)
                SELECT
                    symbol || '_' || data_type,
                    event_timestamp,
                    CURRENT_TIMESTAMP,
                    data_json
                FROM commodities_raw_data
            """)
            migrated = cursor.rowcount
            logger.info(f"✓ commodities_raw_data -> raw_data: 迁移 {migrated} 条")
        else:
            logger.warning("raw_data 表不存在，无法合并 commodities_raw_data")

    # --- 5. 合并 us_factor_data -> factor_data ---
    if 'us_factor_data' in existing_tables:
        if 'factor_data' in existing_tables:
            cursor.execute("""
                INSERT OR IGNORE INTO factor_data (symbol, timestamp, factor_name, factor_value)
                SELECT symbol, timestamp, factor_name, factor_value
                FROM us_factor_data
            """)
            migrated = cursor.rowcount
            logger.info(f"✓ us_factor_data -> factor_data: 迁移 {migrated} 条")
        else:
            logger.warning("factor_data 表不存在，无法合并 us_factor_data")

    # --- 6. 合并 commodities_factor_data -> factor_data ---
    if 'commodities_factor_data' in existing_tables:
        if 'factor_data' in existing_tables:
            cursor.execute("""
                INSERT OR IGNORE INTO factor_data (symbol, timestamp, factor_name, factor_value)
                SELECT symbol, timestamp, factor_name, factor_value
                FROM commodities_factor_data
            """)
            migrated = cursor.rowcount
            logger.info(f"✓ commodities_factor_data -> factor_data: 迁移 {migrated} 条")
        else:
            logger.warning("factor_data 表不存在，无法合并 commodities_factor_data")

    # --- 7. 删除已废弃的旧表 ---
    tables_to_drop = ['us_raw_data', 'us_factor_data', 'commodities_raw_data', 'commodities_factor_data']
    for table in tables_to_drop:
        if table in existing_tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            logger.info(f"✓ 删除旧表: {table}")

    conn.commit()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    final_tables = [row[0] for row in cursor.fetchall()]
    logger.info(f"迁移后表列表: {final_tables}")

    conn.close()
    logger.info(f"========== 迁移完成: {db_path} ==========\n")


def main():
    print("=" * 70)
    print("  多资产数据库重构迁移脚本")
    print("  将同时修改 PROD 和 DEV 数据库")
    print("=" * 70)
    print()
    print("迁移内容:")
    print("  1. us_market_data -> market_data_us_stock")
    print("  2. commodities_market_data -> market_data_commodity")
    print("  3. us_raw_data / commodities_raw_data -> raw_data (合并)")
    print("  4. us_factor_data / commodities_factor_data -> factor_data (合并)")
    print("  5. 删除废弃旧表")
    print()

    confirm = input("确认执行迁移？此操作不可逆！(输入 YES 继续): ")
    if confirm != "YES":
        print("已取消迁移")
        return

    for db_path in [DB_PROD, DB_DEV]:
        try:
            migrate_database(db_path)
        except Exception as e:
            logger.error(f"迁移失败 {db_path}: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ 迁移全部完成！请验证数据库后继续代码重构。")


if __name__ == "__main__":
    main()
