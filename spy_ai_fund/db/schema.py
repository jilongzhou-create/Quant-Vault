#!/usr/bin/env python3
"""
SPY AI Fund - Database Schema

SPY专属表 (spy_ 前缀), 与 gold 的 ai_factor_registry 等表隔离:
  1. spy_ai_factor_registry   - SPY因子注册表
  2. spy_ai_factor_audit      - SPY IS审核记录
  3. spy_ai_factor_oos_autopsy - SPY OOS验尸记录
  4. spy_ai_agent_log         - SPY Agent运行日志
  5. spy_ai_data_requirements - SPY数据需求表
"""

import sqlite3
from spy_ai_fund.config import DB_PATH

FACTOR_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS spy_ai_factor_registry (
    factor_id       TEXT PRIMARY KEY,
    factor_class    TEXT NOT NULL,
    source_file     TEXT NOT NULL,
    mining_direction TEXT NOT NULL,
    mining_method   TEXT NOT NULL,
    llm_prompt_hash TEXT,
    status          TEXT DEFAULT 'draft',
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

FACTOR_AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS spy_ai_factor_audit (
    audit_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_id       TEXT NOT NULL,
    is_start        TEXT NOT NULL,
    is_end          TEXT NOT NULL,
    global_ic       REAL,
    conditional_ic  REAL,
    hit_rate        REAL,
    trigger_rate    REAL,
    max_corr_with   TEXT,
    max_corr_value  REAL,
    verdict         TEXT NOT NULL,
    reject_reason   TEXT,
    audited_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (factor_id) REFERENCES spy_ai_factor_registry(factor_id)
)
"""

FACTOR_OOS_AUTOPSY_DDL = """
CREATE TABLE IF NOT EXISTS spy_ai_factor_oos_autopsy (
    autopsy_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_id       TEXT NOT NULL,
    oos_start       TEXT NOT NULL,
    oos_end         TEXT NOT NULL,
    global_ic       REAL,
    conditional_ic  REAL,
    hit_rate        REAL,
    trigger_rate    REAL,
    verdict         TEXT NOT NULL,
    reject_reason   TEXT,
    autopsied_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (factor_id) REFERENCES spy_ai_factor_registry(factor_id)
)
"""

AGENT_LOG_DDL = """
CREATE TABLE IF NOT EXISTS spy_ai_agent_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name      TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    action          TEXT NOT NULL,
    input_summary   TEXT,
    output_summary  TEXT,
    duration_sec    REAL,
    status          TEXT DEFAULT 'success',
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

DATA_REQUIREMENTS_DDL = """
CREATE TABLE IF NOT EXISTS spy_ai_data_requirements (
    req_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    direction       TEXT NOT NULL,
    method          TEXT NOT NULL,
    required_field  TEXT NOT NULL,
    source_type     TEXT NOT NULL DEFAULT 'fred',
    source_id       TEXT NOT NULL,
    is_optional     INTEGER DEFAULT 0,
    coverage_status TEXT DEFAULT 'unknown',
    checked_at      DATETIME,
    UNIQUE(direction, method, required_field)
)
"""

FACTOR_REGISTRY_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_spy_ai_factor_status ON spy_ai_factor_registry (status)",
    "CREATE INDEX IF NOT EXISTS idx_spy_ai_factor_direction ON spy_ai_factor_registry (mining_direction)",
]

FACTOR_AUDIT_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_spy_ai_audit_factor ON spy_ai_factor_audit (factor_id)",
    "CREATE INDEX IF NOT EXISTS idx_spy_ai_audit_verdict ON spy_ai_factor_audit (verdict)",
]

FACTOR_OOS_AUTOPSY_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_spy_ai_oos_autopsy_factor ON spy_ai_factor_oos_autopsy (factor_id)",
    "CREATE INDEX IF NOT EXISTS idx_spy_ai_oos_autopsy_verdict ON spy_ai_factor_oos_autopsy (verdict)",
]

AGENT_LOG_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_spy_ai_agent_run ON spy_ai_agent_log (run_id)",
    "CREATE INDEX IF NOT EXISTS idx_spy_ai_agent_name ON spy_ai_agent_log (agent_name)",
]

DATA_REQUIREMENTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_spy_ai_data_req_dm ON spy_ai_data_requirements (direction, method)",
    "CREATE INDEX IF NOT EXISTS idx_spy_ai_data_req_status ON spy_ai_data_requirements (coverage_status)",
]

ALL_AI_FUND_DDL = (
    [FACTOR_REGISTRY_DDL, FACTOR_AUDIT_DDL, FACTOR_OOS_AUTOPSY_DDL,
     AGENT_LOG_DDL, DATA_REQUIREMENTS_DDL]
    + FACTOR_REGISTRY_INDEXES
    + FACTOR_AUDIT_INDEXES
    + FACTOR_OOS_AUTOPSY_INDEXES
    + AGENT_LOG_INDEXES
    + DATA_REQUIREMENTS_INDEXES
)


def init_ai_fund_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for ddl in ALL_AI_FUND_DDL:
        cursor.execute(ddl)
    conn.commit()
    conn.close()


def log_agent(run_id, agent_name, action,
              input_summary=None, output_summary=None,
              duration_sec=None, status='success'):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO spy_ai_agent_log
           (agent_name, run_id, action, input_summary, output_summary, duration_sec, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (agent_name, run_id, action, input_summary, output_summary, duration_sec, status)
    )
    conn.commit()
    conn.close()


def register_factor(factor_id, factor_class, source_file,
                    mining_direction, mining_method,
                    llm_prompt_hash=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT OR IGNORE INTO spy_ai_factor_registry
           (factor_id, factor_class, source_file, mining_direction, mining_method, llm_prompt_hash, status)
           VALUES (?, ?, ?, ?, ?, ?, 'draft')""",
        (factor_id, factor_class, source_file, mining_direction, mining_method, llm_prompt_hash)
    )
    conn.commit()
    conn.close()


def update_factor_status(factor_id, status):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE spy_ai_factor_registry SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE factor_id = ?",
        (status, factor_id)
    )
    conn.commit()
    conn.close()


def save_audit(factor_id, is_start, is_end,
               global_ic, conditional_ic,
               hit_rate, trigger_rate,
               max_corr_with, max_corr_value,
               verdict, reject_reason=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO spy_ai_factor_audit
           (factor_id, is_start, is_end, global_ic, conditional_ic, hit_rate,
            trigger_rate, max_corr_with, max_corr_value, verdict, reject_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (factor_id, is_start, is_end, global_ic, conditional_ic, hit_rate,
         trigger_rate, max_corr_with, max_corr_value, verdict, reject_reason)
    )
    conn.commit()
    conn.close()


def get_accepted_factors():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT factor_id, factor_class, source_file, mining_direction, mining_method "
        "FROM spy_ai_factor_registry WHERE status = 'accepted'"
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            'factor_id': r[0], 'factor_class': r[1], 'source_file': r[2],
            'mining_direction': r[3], 'mining_method': r[4],
        }
        for r in rows
    ]


def get_factor_signals_for_orthogonality():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT factor_id, source_file FROM spy_ai_factor_registry WHERE status = 'accepted'"
    )
    rows = cursor.fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def save_oos_autopsy(factor_id, oos_start, oos_end,
                     global_ic, conditional_ic,
                     hit_rate, trigger_rate,
                     verdict, reject_reason=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO spy_ai_factor_oos_autopsy
           (factor_id, oos_start, oos_end, global_ic, conditional_ic, hit_rate,
            trigger_rate, verdict, reject_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (factor_id, oos_start, oos_end, global_ic, conditional_ic, hit_rate,
         trigger_rate, verdict, reject_reason)
    )
    conn.commit()
    conn.close()


def get_factors_needing_oos_autopsy():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT factor_id, factor_class, source_file, mining_direction, mining_method "
        "FROM spy_ai_factor_registry "
        "WHERE status = 'is_passed' "
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            'factor_id': r[0], 'factor_class': r[1], 'source_file': r[2],
            'mining_direction': r[3], 'mining_method': r[4],
        }
        for r in rows
    ]


def check_data_feasibility(direction, method, available_fields):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT required_field, source_type, source_id, is_optional "
        "FROM spy_ai_data_requirements WHERE direction = ? AND method = ?",
        (direction, method)
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {
            'feasible': True,
            'missing_required': [],
            'missing_optional': [],
            'auto_fetchable': [],
        }

    missing_required = []
    missing_optional = []
    auto_fetchable = []

    for field, src_type, src_id, is_opt in rows:
        if field in available_fields:
            continue
        if is_opt:
            missing_optional.append(field)
        else:
            missing_required.append(field)
            if src_type == 'fred' and src_id:
                auto_fetchable.append({
                    'field': field,
                    'source_type': src_type,
                    'source_id': src_id,
                })

    feasible = len(missing_required) == 0 or len(auto_fetchable) == len(missing_required)

    return {
        'feasible': feasible,
        'missing_required': missing_required,
        'missing_optional': missing_optional,
        'auto_fetchable': auto_fetchable,
    }


def update_data_coverage_status(direction, method, required_field, status):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE spy_ai_data_requirements SET coverage_status = ?, checked_at = CURRENT_TIMESTAMP "
        "WHERE direction = ? AND method = ? AND required_field = ?",
        (status, direction, method, required_field)
    )
    conn.commit()
    conn.close()


DATA_REQUIREMENTS_SEED = [
    {'direction': 'policy_pivot', 'method': 'unstructured', 'required_field': 'fomc_sentiment',
     'source_type': 'internal', 'source_id': 'fomc_sentiment', 'is_optional': 0},
    {'direction': 'policy_pivot', 'method': 'nonlinear', 'required_field': 't10y2y',
     'source_type': 'fred', 'source_id': 'T10Y2Y', 'is_optional': 0},
    {'direction': 'policy_pivot', 'method': 'nonlinear', 'required_field': 'dgs10',
     'source_type': 'fred', 'source_id': 'DGS10', 'is_optional': 1},
    {'direction': 'policy_pivot', 'method': 'nonlinear', 'required_field': 'dgs2',
     'source_type': 'fred', 'source_id': 'DGS2', 'is_optional': 1},
    {'direction': 'panic_mean_reversion', 'method': 'nonlinear', 'required_field': 'vixcls',
     'source_type': 'fred', 'source_id': 'VIXCLS', 'is_optional': 0},
    {'direction': 'panic_mean_reversion', 'method': 'nonlinear', 'required_field': 'bamlh0a0hym2',
     'source_type': 'fred', 'source_id': 'BAMLH0A0HYM2', 'is_optional': 0},
    {'direction': 'panic_mean_reversion', 'method': 'nonlinear', 'required_field': 'baa10ym',
     'source_type': 'fred', 'source_id': 'BAA10YM', 'is_optional': 1},
    {'direction': 'panic_mean_reversion', 'method': 'unstructured', 'required_field': 'fomc_sentiment',
     'source_type': 'internal', 'source_id': 'fomc_sentiment', 'is_optional': 0},
]


def seed_data_requirements():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for req in DATA_REQUIREMENTS_SEED:
        cursor.execute(
            """INSERT OR IGNORE INTO spy_ai_data_requirements
               (direction, method, required_field, source_type, source_id, is_optional)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (req['direction'], req['method'], req['required_field'],
             req['source_type'], req['source_id'], req['is_optional'])
        )
    conn.commit()
    conn.close()
