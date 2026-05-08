#!/usr/bin/env python3

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

_dotenv_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(_dotenv_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(_dotenv_path)
    except ImportError:
        with open(_dotenv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value

IS_START = '2007-01-01'
IS_END = '2019-12-31'
OOS_START = '2020-01-01'
OOS_END = '2026-04-30'

DTB3_START_DATE = '2007-01-01'

SPY_FRED_SERIES = [
    ("INDPRO",    "Industrial Production Index"),
    ("ICSA",      "Initial Unemployment Claims"),
    ("WALCL",     "Fed Total Assets"),
    ("WTREGEN",   "Treasury General Account"),
    ("RRPONTSYD", "Overnight Reverse Repo"),
    ("DTB3",      "3-Month Treasury Bill Rate"),
]

GLOBAL_IC_THRESHOLD = 0.02
COND_IC_THRESHOLD = 0.05
HIT_RATE_THRESHOLD = 0.55
TRIGGER_RATE_THRESHOLD = 0.001
TRIGGER_RATE_CAP = 0.20

MARGINAL_SHARPE_THRESHOLD_NORMAL = 0.005
MARGINAL_SHARPE_THRESHOLD_ELITE = 0.001
MARGINAL_ELITE_COND_IC = 0.20

ZSCORE_WINDOW = int(os.environ.get('ZSCORE_WINDOW', '252'))
ZSCORE_THRESHOLD = float(os.environ.get('ZSCORE_THRESHOLD', '2.5'))
SAT_ASYM_CAP = float(os.environ.get('SAT_ASYM_CAP', '0.3'))
CORE_BULL_PROTECT = float(os.environ.get('CORE_BULL_PROTECT', '0.75'))

CORE_FORBIDDEN_COLUMNS = {'indpro', 'icsa', 'walcl', 'wtregen', 'rrpontsyd'}

MINING_DIRECTIONS = ['policy_pivot', 'panic_mean_reversion']
MINING_METHODS = ['unstructured', 'nonlinear']
DISABLED_METHODS = set()

FRED_API_KEY = os.environ.get('FRED_API_KEY', '')

GCP_PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'synthmind-social-content')
GCP_LOCATION = os.environ.get('GCP_LOCATION', 'global')
GOOGLE_APPLICATION_CREDENTIALS = os.path.join(PROJECT_ROOT, 'vertex_key.json')
LLM_MODEL = 'gemini-3.1-pro-preview'
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096

DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'trading_system_prod.db')

FACTORS_DIR = os.path.join(os.path.dirname(__file__), 'factors')

AUTO_MINE_INTERVAL_SEC = int(os.environ.get('AUTO_MINE_INTERVAL_SEC', '30'))
AUTO_MINE_MAX_CYCLES = int(os.environ.get('AUTO_MINE_MAX_CYCLES', '0'))
