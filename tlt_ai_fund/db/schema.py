#!/usr/bin/env python3
"""
TLT AI Fund - 数据库 Schema

五张表, 使用 tlt_ai_ 前缀, 与 gold/spy/btc 的表完全隔离:
  1. tlt_ai_factor_registry    - 因子注册表
  2. tlt_ai_factor_audit       - IS 审核记录
  3. tlt_ai_factor_oos_autopsy - OOS 验尸记录
  4. tlt_ai_agent_log          - Agent 运行日志
  5. tlt_ai_data_requirements  - 数据需求表 (TLT/FICC 专属种子)
"""

import sqlite3
from tlt_ai_fund.config import DB_PATH

PREFIX = 'tlt_ai_'

FACTOR_REGISTRY_DDL = f"""
CREATE TABLE IF NOT EXISTS {PREFIX}factor_registry (
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

FACTOR_AUDIT_DDL = f"""
CREATE TABLE IF NOT EXISTS {PREFIX}factor_audit (
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
    FOREIGN KEY (factor_id) REFERENCES {PREFIX}factor_registry(factor_id)
)
"""

FACTOR_OOS_AUTOPSY_DDL = f"""
CREATE TABLE IF NOT EXISTS {PREFIX}factor_oos_autopsy (
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
    FOREIGN KEY (factor_id) REFERENCES {PREFIX}factor_registry(factor_id)
)
"""

AGENT_LOG_DDL = f"""
CREATE TABLE IF NOT EXISTS {PREFIX}agent_log (
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

DATA_REQUIREMENTS_DDL = f"""
CREATE TABLE IF NOT EXISTS {PREFIX}data_requirements (
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

DATA_REQUIREMENTS_INDEXES = [
    f"CREATE INDEX IF NOT EXISTS idx_{PREFIX}data_req_dm ON {PREFIX}data_requirements (direction, method)",
    f"CREATE INDEX IF NOT EXISTS idx_{PREFIX}data_req_status ON {PREFIX}data_requirements (coverage_status)",
]

FACTOR_REGISTRY_INDEXES = [
    f"CREATE INDEX IF NOT EXISTS idx_{PREFIX}factor_status ON {PREFIX}factor_registry (status)",
    f"CREATE INDEX IF NOT EXISTS idx_{PREFIX}factor_direction ON {PREFIX}factor_registry (mining_direction)",
]

FACTOR_AUDIT_INDEXES = [
    f"CREATE INDEX IF NOT EXISTS idx_{PREFIX}audit_factor ON {PREFIX}factor_audit (factor_id)",
    f"CREATE INDEX IF NOT EXISTS idx_{PREFIX}audit_verdict ON {PREFIX}factor_audit (verdict)",
]

FACTOR_OOS_AUTOPSY_INDEXES = [
    f"CREATE INDEX IF NOT EXISTS idx_{PREFIX}oos_autopsy_factor ON {PREFIX}factor_oos_autopsy (factor_id)",
    f"CREATE INDEX IF NOT EXISTS idx_{PREFIX}oos_autopsy_verdict ON {PREFIX}factor_oos_autopsy (verdict)",
]

AGENT_LOG_INDEXES = [
    f"CREATE INDEX IF NOT EXISTS idx_{PREFIX}agent_run ON {PREFIX}agent_log (run_id)",
    f"CREATE INDEX IF NOT EXISTS idx_{PREFIX}agent_name ON {PREFIX}agent_log (agent_name)",
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


def log_agent(run_id: str, agent_name: str, action: str,
              input_summary: str = None, output_summary: str = None,
              duration_sec: float = None, status: str = 'success'):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        f"""INSERT INTO {PREFIX}agent_log
           (agent_name, run_id, action, input_summary, output_summary, duration_sec, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (agent_name, run_id, action, input_summary, output_summary, duration_sec, status)
    )
    conn.commit()
    conn.close()


def register_factor(factor_id: str, factor_class: str, source_file: str,
                    mining_direction: str, mining_method: str,
                    llm_prompt_hash: str = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        f"""INSERT OR IGNORE INTO {PREFIX}factor_registry
           (factor_id, factor_class, source_file, mining_direction, mining_method, llm_prompt_hash, status)
           VALUES (?, ?, ?, ?, ?, ?, 'draft')""",
        (factor_id, factor_class, source_file, mining_direction, mining_method, llm_prompt_hash)
    )
    conn.commit()
    conn.close()


def update_factor_status(factor_id: str, status: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        f"UPDATE {PREFIX}factor_registry SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE factor_id = ?",
        (status, factor_id)
    )
    conn.commit()
    conn.close()


def save_audit(factor_id: str, is_start: str, is_end: str,
               global_ic: float, conditional_ic: float,
               hit_rate: float, trigger_rate: float,
               max_corr_with: str, max_corr_value: float,
               verdict: str, reject_reason: str = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        f"""INSERT INTO {PREFIX}factor_audit
           (factor_id, is_start, is_end, global_ic, conditional_ic, hit_rate,
            trigger_rate, max_corr_with, max_corr_value, verdict, reject_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (factor_id, is_start, is_end, global_ic, conditional_ic, hit_rate,
         trigger_rate, max_corr_with, max_corr_value, verdict, reject_reason)
    )
    conn.commit()
    conn.close()


def get_accepted_factors() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT factor_id, factor_class, source_file, mining_direction, mining_method "
        f"FROM {PREFIX}factor_registry WHERE status = 'accepted'"
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


def get_factor_signals_for_orthogonality() -> dict[str, str]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT factor_id, source_file FROM {PREFIX}factor_registry WHERE status = 'accepted'"
    )
    rows = cursor.fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def save_oos_autopsy(factor_id: str, oos_start: str, oos_end: str,
                     global_ic: float, conditional_ic: float,
                     hit_rate: float, trigger_rate: float,
                     verdict: str, reject_reason: str = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        f"""INSERT INTO {PREFIX}factor_oos_autopsy
           (factor_id, oos_start, oos_end, global_ic, conditional_ic, hit_rate,
            trigger_rate, verdict, reject_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (factor_id, oos_start, oos_end, global_ic, conditional_ic, hit_rate,
         trigger_rate, verdict, reject_reason)
    )
    conn.commit()
    conn.close()


def get_factors_needing_oos_autopsy() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT r.factor_id, r.factor_class, r.source_file, r.mining_direction, r.mining_method "
        f"FROM {PREFIX}factor_registry r "
        f"WHERE r.status = 'is_passed' "
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


DATA_REQUIREMENTS_SEED = [
    {'direction': 'unstructured', 'method': 'unstructured', 'required_field': 'fomc_sentiment',
     'source_type': 'internal', 'source_id': 'fomc_sentiment', 'is_optional': 0},
    {'direction': 'unstructured', 'method': 'nonlinear', 'required_field': 'fomc_sentiment',
     'source_type': 'internal', 'source_id': 'fomc_sentiment', 'is_optional': 0},
    {'direction': 'unstructured', 'method': 'nonlinear', 'required_field': 'vixcls',
     'source_type': 'fred', 'source_id': 'VIXCLS', 'is_optional': 1},
    {'direction': 'microstructure', 'method': 'nonlinear', 'required_field': 'vixcls',
     'source_type': 'fred', 'source_id': 'VIXCLS', 'is_optional': 0},
    {'direction': 'microstructure', 'method': 'nonlinear', 'required_field': 'nfci',
     'source_type': 'fred', 'source_id': 'NFCI', 'is_optional': 1},
    {'direction': 'microstructure', 'method': 'nonlinear', 'required_field': 'stlfsi4',
     'source_type': 'fred', 'source_id': 'STLFSI4', 'is_optional': 1},
    {'direction': 'microstructure', 'method': 'options', 'required_field': 'vixcls',
     'source_type': 'fred', 'source_id': 'VIXCLS', 'is_optional': 0},
    {'direction': 'microstructure', 'method': 'options', 'required_field': 'gvzcls',
     'source_type': 'fred', 'source_id': 'GVZCLS', 'is_optional': 1},
    {'direction': 'microstructure', 'method': 'unstructured', 'required_field': 'fomc_sentiment',
     'source_type': 'internal', 'source_id': 'fomc_sentiment', 'is_optional': 0},
    {'direction': 'volatility', 'method': 'nonlinear', 'required_field': 'vixcls',
     'source_type': 'fred', 'source_id': 'VIXCLS', 'is_optional': 0},
    {'direction': 'volatility', 'method': 'nonlinear', 'required_field': 't10y2y',
     'source_type': 'fred', 'source_id': 'T10Y2Y', 'is_optional': 1},
    {'direction': 'volatility', 'method': 'nonlinear', 'required_field': 'usepuindxd',
     'source_type': 'fred', 'source_id': 'USEPUINDXD', 'is_optional': 1},
    {'direction': 'volatility', 'method': 'options', 'required_field': 'vixcls',
     'source_type': 'fred', 'source_id': 'VIXCLS', 'is_optional': 0},
    {'direction': 'volatility', 'method': 'options', 'required_field': 'gvzcls',
     'source_type': 'fred', 'source_id': 'GVZCLS', 'is_optional': 1},
    {'direction': 'volatility', 'method': 'unstructured', 'required_field': 'fomc_sentiment',
     'source_type': 'internal', 'source_id': 'fomc_sentiment', 'is_optional': 0},
]


def seed_data_requirements():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for req in DATA_REQUIREMENTS_SEED:
        cursor.execute(
            f"""INSERT OR IGNORE INTO {PREFIX}data_requirements
               (direction, method, required_field, source_type, source_id, is_optional)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (req['direction'], req['method'], req['required_field'],
             req['source_type'], req['source_id'], req['is_optional'])
        )
    conn.commit()
    conn.close()


def check_data_feasibility(direction: str, method: str,
                           available_fields: set) -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT required_field, source_type, source_id, is_optional "
        f"FROM {PREFIX}data_requirements WHERE direction = ? AND method = ?",
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


def update_data_coverage_status(direction: str, method: str,
                                 required_field: str, status: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        f"UPDATE {PREFIX}data_requirements SET coverage_status = ?, checked_at = CURRENT_TIMESTAMP "
        f"WHERE direction = ? AND method = ? AND required_field = ?",
        (status, direction, method, required_field)
    )
    conn.commit()
    conn.close()


def get_legacy_draft_factors(exclude_ids: list) -> list[dict]:
    exclude_set = set(exclude_ids)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT factor_id, source_file FROM {PREFIX}factor_registry "
        f"WHERE status = 'draft'"
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {'factor_id': r[0], 'source_file': r[1]}
        for r in rows if r[0] not in exclude_set
    ]
