#!/usr/bin/env python3
"""
将 NEEDS_DISCUSSION 状态的数据需求标记为 COMPLETED，
并唤醒因数据就绪而解冻的冷冻策略。

支持两种模式：
  1. 默认模式：将所有 NEEDS_DISCUSSION 标记为 COMPLETED
  2. 指定 ID 模式：只将指定 ID 的需求标记为 COMPLETED
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sqlite3
from config import DB_PATH
from database.db_manager import awaken_completed_strategies
from logger_setup import get_logger

logger = get_logger(__name__)

RESOLVED_IDS = [242, 243, 244, 245, 247]


def resolve_needs_discussion(req_ids=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if req_ids:
        placeholders = ','.join('?' for _ in req_ids)
        cursor.execute(f"SELECT id, source_name, status FROM data_requirements WHERE id IN ({placeholders})", req_ids)
        rows = cursor.fetchall()
        print(f"指定需求数: {len(rows)}")
        for r in rows:
            print(f"  ID={r[0]}, source={r[1]}, status={r[2]}")

        cursor.execute(f"UPDATE data_requirements SET status = 'COMPLETED' WHERE id IN ({placeholders}) AND status = 'NEEDS_DISCUSSION'", req_ids)
        updated = cursor.rowcount
    else:
        cursor.execute("SELECT COUNT(*) FROM data_requirements WHERE status = 'NEEDS_DISCUSSION'")
        count = cursor.fetchone()[0]
        print(f"当前 NEEDS_DISCUSSION 需求数: {count}")

        if count == 0:
            print("无需更新")
            conn.close()
            return

        cursor.execute("UPDATE data_requirements SET status = 'COMPLETED' WHERE status = 'NEEDS_DISCUSSION'")
        updated = cursor.rowcount

    conn.commit()
    conn.close()

    print(f"已将 {updated} 条需求更新为 COMPLETED")
    logger.info(f"resolve_data_requirements: {updated} 条需求标记为 COMPLETED")

    awakened = awaken_completed_strategies()
    if awakened:
        print(f"成功唤醒 {len(awakened)} 个冷冻策略: {awakened}")
    else:
        print("没有需要唤醒的冷冻策略")


if __name__ == "__main__":
    resolve_needs_discussion(req_ids=RESOLVED_IDS)
