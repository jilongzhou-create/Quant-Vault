"""
SaaS 平台独立配置模块 (统一配置网关)

设计原则：
  1. 本模块是 SaaS 平台唯一的配置入口，所有模块禁止直接 os.getenv / os.environ.get
  2. 双模式兼容：
     - 本地开发：从项目根目录 .env 文件加载
     - 云端部署：直接读取环境变量（Docker -e / K8s Secret / 云平台面板）
  3. 所有 API Key 统一在此声明，支持 SAAS_ 前缀映射
     - 本地 .env 中可写 SAAS_FRED_API_KEY 或 FRED_API_KEY（优先 SAAS_ 前缀）
     - 云端环境变量同理
  4. 严禁引入本地投研系统的 DB_PATH / BINANCE_API_KEY 等配置
"""

import os
import sys
import logging
from dotenv import load_dotenv

_saas_root = os.path.abspath(os.path.dirname(__file__))
_project_root = os.path.abspath(os.path.join(_saas_root, '..'))

_env_file = os.path.join(_project_root, '.env')
if os.path.exists(_env_file):
    load_dotenv(_env_file)


def _get_key(saas_key: str, fallback_key: str = '', default: str = '') -> str:
    """
    优先读取 SAAS_ 前缀的环境变量，fallback 到无前缀版本

    例如: _get_key('SAAS_FRED_API_KEY', 'FRED_API_KEY')
      -> 优先读 SAAS_FRED_API_KEY，不存在则读 FRED_API_KEY，都没有返回 ''
    """
    val = os.environ.get(saas_key, '')
    if val:
        return val
    if fallback_key:
        val = os.environ.get(fallback_key, '')
        if val:
            return val
    return default


# ============================================================
# Supabase 云端数据库
# ============================================================
SAAS_SUPABASE_URL = _get_key('SAAS_SUPABASE_URL')
SAAS_SUPABASE_KEY = _get_key('SAAS_SUPABASE_KEY')

# ============================================================
# 数据源 API Keys
# ============================================================
FRED_API_KEY = _get_key('SAAS_FRED_API_KEY', 'FRED_API_KEY')
FMP_API_KEY = _get_key('SAAS_FMP_API_KEY', 'FMP_API_KEY')
COINMETRICS_API_KEY = _get_key('SAAS_COINMETRICS_API_KEY', 'COINMETRICS_API_KEY')

# ============================================================
# AI 大模型 API Keys
# ============================================================
GEMINI_API_KEY = _get_key('SAAS_GEMINI_API_KEY', 'GEMINI_API_KEY')

# ============================================================
# 安全加密主密钥 (Fernet 对称加密，用于加密用户 API Key)
# 生成方式: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# ============================================================
MASTER_KEY = _get_key('SAAS_MASTER_KEY', 'MASTER_KEY')

# ============================================================
# 代理配置 (云端通常不需要，但保留兼容)
# ============================================================
PROXY_MODE = _get_key('SAAS_PROXY_MODE', 'PROXY_MODE', 'direct').lower().strip()
PROXY_URL = _get_key('SAAS_PROXY_URL', 'HTTPS_PROXY') or _get_key('SAAS_PROXY_URL', 'HTTP_PROXY')

# ============================================================
# 运行环境
# ============================================================
SAAS_ENV = _get_key('SAAS_ENV', default='development').lower()
SAAS_LOG_LEVEL = logging.DEBUG if SAAS_ENV == 'development' else logging.INFO

_saas_logger = logging.getLogger('saas_platform')
_saas_logger.setLevel(SAAS_LOG_LEVEL)

_missing = []
if not SAAS_SUPABASE_URL:
    _missing.append('SAAS_SUPABASE_URL')
if not SAAS_SUPABASE_KEY:
    _missing.append('SAAS_SUPABASE_KEY')
if _missing:
    _saas_logger.warning(f"未配置: {', '.join(_missing)}，云端数据库功能不可用")


def get_saas_config(key: str, default: str = '') -> str:
    """
    通用配置读取接口

    支持两种 key 格式：
      - SAAS_ 前缀：直接从 _saas_config 读取
      - 非 SAAS_ 前缀：自动映射为 SAAS_ 前缀 + fallback 无前缀
    """
    saas_key = key if key.startswith('SAAS_') else f'SAAS_{key}'
    fallback = key if not key.startswith('SAAS_') else key[5:]
    return _get_key(saas_key, fallback, default)


def is_configured() -> bool:
    return bool(SAAS_SUPABASE_URL and SAAS_SUPABASE_KEY)


def get_config_summary() -> dict:
    """
    返回当前配置状态摘要（隐藏敏感值），用于启动日志和健康检查
    """
    def _mask(val: str) -> str:
        if not val:
            return '<未配置>'
        if len(val) <= 8:
            return '***'
        return val[:4] + '***' + val[-4:]

    return {
        'SAAS_ENV': SAAS_ENV,
        'SAAS_SUPABASE_URL': _mask(SAAS_SUPABASE_URL),
        'SAAS_SUPABASE_KEY': _mask(SAAS_SUPABASE_KEY),
        'FRED_API_KEY': _mask(FRED_API_KEY),
        'FMP_API_KEY': _mask(FMP_API_KEY),
        'COINMETRICS_API_KEY': _mask(COINMETRICS_API_KEY),
        'GEMINI_API_KEY': _mask(GEMINI_API_KEY),
        'MASTER_KEY': _mask(MASTER_KEY),
        'PROXY_MODE': PROXY_MODE,
    }
