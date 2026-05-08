#!/usr/bin/env python3
"""
TLT 数据同步脚本

功能:
  1. 从 FMP API 抓取 TLT 历史日线数据 (adjClose 为计算基准)
  2. 确保 DTB3 (3个月美债收益率) FRED 数据就绪
  3. 初始化 TLT 数据库表

用法:
  python tlt_ai_fund/sync_tlt_data.py              # 增量同步
  python tlt_ai_fund/sync_tlt_data.py --full      # 全量同步
"""

import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logger_setup import get_logger
from data_pipeline.fetchers.tlt_market_fetcher import fetch_and_store_tlt
from database.db_manager import add_fred_series, get_fred_series_ids

logger = get_logger(__name__)


def ensure_dtb3_series():
    """确保 DTB3 FRED 系列已配置"""
    DTB3_SERIES = [("DTB3", "3-Month Treasury Bill: Secondary Market Rate")]

    for series_id, title in DTB3_SERIES:
        existing = get_fred_series_ids()
        if series_id not in existing:
            added = add_fred_series(series_id, title, category='core')
            if added:
                logger.info(f"[FRED] 新增 DTB3 系列配置: {series_id}")
            else:
                logger.info(f"[FRED] DTB3 系列已存在: {series_id}")
        else:
            logger.info(f"[FRED] DTB3 系列已配置: {series_id}")


def init_tlt_table():
    """初始化 TLT 数据库表"""
    import sqlite3
    from database.db_manager import DB_PATH

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_data_tlt (
        symbol          TEXT        NOT NULL,
        timestamp       DATETIME    NOT NULL,
        date            TEXT        NOT NULL,
        open            REAL,
        high            REAL,
        low             REAL,
        close           REAL,
        adj_close       REAL        NOT NULL,
        volume          REAL,
        rsi_14          REAL,
        macd            REAL,
        macd_signal     REAL,
        macd_hist       REAL,
        PRIMARY KEY (symbol, timestamp)
    )
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tlt_timestamp ON market_data_tlt (timestamp DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tlt_date ON market_data_tlt (date)')

    conn.commit()
    conn.close()
    logger.info("[DB] TLT market_data_tlt 表初始化完成")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='TLT Data Sync')
    parser.add_argument('--full', action='store_true', help='全量同步 TLT 数据')
    parser.add_argument('--start', type=str, default='2007-01-01', help='起始日期')
    args = parser.parse_args()

    print("=" * 60)
    print("  TLT Data Sync Pipeline")
    print("=" * 60)

    print("\n[Step 1/3] 初始化 TLT 数据库表...")
    init_tlt_table()

    print("\n[Step 2/3] 确保 DTB3 FRED 系列配置...")
    ensure_dtb3_series()

    print("\n[Step 3/3] 抓取 TLT 市场数据 (使用 adjClose)...")
    count = fetch_and_store_tlt(start_date=args.start, full_sync=args.full)

    print(f"\n{'='*60}")
    print(f"  TLT 数据同步完成！共保存 {count} 条记录")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
