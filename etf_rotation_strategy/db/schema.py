#!/usr/bin/env python3
"""
ETF 轮动策略 - 数据库表结构定义

因子生命周期 (当前简化版):
  draft → is_passed → accepted

预留完整生命周期 (后续 Phase 扩展):
  draft → is_passed → accepted → dead / dead_oos / superseded
"""

import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'etf_rotation.db'
)


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_tables(db_path: str = DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info(f"[DB] 表初始化完成: {db_path}")
    finally:
        conn.close()


SCHEMA_SQL = """
-- 1. 因子注册表
-- 生命周期: draft → is_passed → accepted (当前)
-- 预留扩展: → dead / dead_oos / superseded
CREATE TABLE IF NOT EXISTS etf_factor_registry (
    factor_id       TEXT PRIMARY KEY,
    formula         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'draft',
    mining_round    INTEGER,
    rank_ic         REAL,
    icir            REAL,
    turnover        REAL,
    max_corr_with   TEXT,
    max_corr_value  REAL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. IS 审核记录
CREATE TABLE IF NOT EXISTS etf_factor_audit (
    audit_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_id       TEXT NOT NULL,
    is_start        TEXT NOT NULL,
    is_end          TEXT NOT NULL,
    rank_ic         REAL,
    icir            REAL,
    turnover        REAL,
    max_corr_with   TEXT,
    max_corr_value  REAL,
    verdict         TEXT NOT NULL,
    reject_reason   TEXT,
    audited_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (factor_id) REFERENCES etf_factor_registry(factor_id)
);

-- 3. OOS 验尸记录 (预留，Phase 2 启用)
CREATE TABLE IF NOT EXISTS etf_factor_oos_autopsy (
    autopsy_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_id       TEXT NOT NULL,
    oos_start       TEXT NOT NULL,
    oos_end         TEXT NOT NULL,
    rank_ic         REAL,
    icir            REAL,
    turnover        REAL,
    verdict         TEXT NOT NULL,
    reject_reason   TEXT,
    autopsied_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (factor_id) REFERENCES etf_factor_registry(factor_id)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_etf_factor_status ON etf_factor_registry (status);
CREATE INDEX IF NOT EXISTS idx_etf_factor_formula ON etf_factor_registry (formula);
CREATE INDEX IF NOT EXISTS idx_etf_audit_factor ON etf_factor_audit (factor_id);
CREATE INDEX IF NOT EXISTS idx_etf_audit_verdict ON etf_factor_audit (verdict);
CREATE INDEX IF NOT EXISTS idx_etf_oos_factor ON etf_factor_oos_autopsy (factor_id);
"""
