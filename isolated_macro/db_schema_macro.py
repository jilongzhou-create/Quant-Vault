#!/usr/bin/env python3
"""
宏观估值模型 - 数据库 Schema 定义

与主系统 db_manager.py 保持一致，采用原生 SQL (sqlite3) 定义表结构。
两张新表完全隔离于主系统的 16 张业务表，互不干扰。

表 1: macro_model_registry - 模型注册表
    存储不同标的估值模型的 β 系数（IS 计算出的"历史物理常数"）

表 2: macro_valuation_daily - 每日估值记录
    存储模型生成的每日理论价值、偏差 Z-Score 和目标敞口
"""

MACRO_MODEL_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS macro_model_registry (
    model_id        TEXT        PRIMARY KEY,
    target_symbol   TEXT        NOT NULL,
    formula_desc    TEXT,
    params_json     TEXT        NOT NULL,
    is_start_date   TEXT,
    is_end_date     TEXT,
    created_at      DATETIME    DEFAULT CURRENT_TIMESTAMP
)
"""

MACRO_VALUATION_DAILY_DDL = """
CREATE TABLE IF NOT EXISTS macro_valuation_daily (
    timestamp       DATETIME    NOT NULL,
    model_id        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    market_price    REAL,
    fair_value      REAL,
    valuation_spread REAL,
    spread_zscore   REAL,
    target_exposure REAL,
    PRIMARY KEY (timestamp, model_id, symbol),
    FOREIGN KEY (model_id) REFERENCES macro_model_registry(model_id)
)
"""

MACRO_VALUATION_DAILY_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_macro_valuation_ts ON macro_valuation_daily (timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_macro_valuation_symbol ON macro_valuation_daily (symbol, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_macro_valuation_model ON macro_valuation_daily (model_id, timestamp DESC)",
]

ALL_MACRO_DDL = [MACRO_MODEL_REGISTRY_DDL, MACRO_VALUATION_DAILY_DDL] + MACRO_VALUATION_DAILY_INDEXES
