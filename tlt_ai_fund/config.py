#!/usr/bin/env python3
"""
TLT AI Fund - 全局配置

核心铁律:
  - IS/OOS 边界锁死, 任何因子评估只能在 IS 内进行
  - Core Anchor 数据 (DFII10/DGS10/BAMLH0A0HYM2) 禁止被卫星因子引用
  - Conditional IC + Hit Rate 双门控, 稀疏数据因子不被误杀
  - 所有卫星因子必须是极端异动脉冲 (Z-Score > 2.5 触发, 其余时间 0.0)
"""

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

FORWARD_PERIOD = 20

GLOBAL_IC_THRESHOLD = 0.02
COND_IC_THRESHOLD = 0.05
HIT_RATE_THRESHOLD = 0.55
TRIGGER_RATE_THRESHOLD = 0.001
TRIGGER_RATE_CAP = 0.20

MARGINAL_SHARPE_THRESHOLD_NORMAL = 0.005
MARGINAL_SHARPE_THRESHOLD_ELITE = 0.001
MARGINAL_ELITE_COND_IC = 0.20

ORTHOGONALITY_THRESHOLD = 0.7

ZSCORE_WINDOW = int(os.environ.get('ZSCORE_WINDOW', '252'))
ZSCORE_THRESHOLD = float(os.environ.get('ZSCORE_THRESHOLD', '2.5'))
SAT_ASYM_CAP = float(os.environ.get('SAT_ASYM_CAP', '0.3'))
CORE_BULL_PROTECT = float(os.environ.get('CORE_BULL_PROTECT', '0.5'))

CORE_FORBIDDEN_COLUMNS = {'dfii10', 'dgs10', 'bamlh0a0hym2'}

MINING_DIRECTIONS = ['unstructured', 'microstructure', 'volatility']
MINING_METHODS = ['unstructured', 'options', 'nonlinear']
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
