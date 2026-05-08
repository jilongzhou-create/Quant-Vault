#!/usr/bin/env python3
"""
SPY Data Sync Pipeline

功能:
  1. 从 FMP API 抓取 SPY 历史日线数据 (adjClose 为计算基准)
  2. 确保 SPY 所需的 FRED 数据系列已配置 (INDPRO, ICSA, WALCL, WTREGEN, RRPONTSYD, DTB3)
  3. 初始化 SPY 数据库表

用法:
  python spy_ai_fund/sync_spy_data.py              # 增量同步
  python spy_ai_fund/sync_spy_data.py --full       # 全量同步
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logger_setup import get_logger
from data_pipeline.fetchers.spy_market_fetcher import fetch_and_store_spy
from database.db_manager import add_fred_series, get_fred_series_ids
from spy_ai_fund.config import SPY_FRED_SERIES

logger = get_logger(__name__)


def ensure_spy_fred_series():
    for series_id, title in SPY_FRED_SERIES:
        existing = get_fred_series_ids()
        if series_id not in existing:
            added = add_fred_series(series_id, title, category='core')
            if added:
                logger.info(f"[FRED] 新增 SPY 底座系列: {series_id}")
            else:
                logger.info(f"[FRED] SPY 底座系列已存在: {series_id}")
        else:
            logger.info(f"[FRED] SPY 底座系列已配置: {series_id}")


def init_spy_table():
    import sqlite3
    from config import DB_PATH

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_data_spy (
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

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_spy_timestamp ON market_data_spy (timestamp DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_spy_date ON market_data_spy (date)')

    conn.commit()
    conn.close()
    logger.info("[DB] SPY market_data_spy 表初始化完成")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SPY Data Sync')
    parser.add_argument('--full', action='store_true', help='全量同步 SPY 数据')
    parser.add_argument('--start', type=str, default='2007-01-01', help='起始日期')
    args = parser.parse_args()

    print("=" * 60)
    print("  SPY Data Sync Pipeline")
    print("=" * 60)

    print("\n[Step 1/3] 初始化 SPY 数据库表...")
    init_spy_table()

    print("\n[Step 2/3] 确保 SPY FRED 系列配置...")
    ensure_spy_fred_series()

    print("\n[Step 3/3] 抓取 SPY 市场数据 (使用 adjClose)...")
    count = fetch_and_store_spy(start_date=args.start, full_sync=args.full)

    print(f"\n{'='*60}")
    print(f"  SPY 数据同步完成！共保存 {count} 条记录")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
