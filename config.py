import os
import sys
import logging
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 添加当前目录到系统路径，解决相对导入问题
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 【环境隔离配置】
# 读取环境变量 SYSTEM_ENV，默认值为 'PROD'
SYSTEM_ENV = os.getenv('SYSTEM_ENV', 'PROD').upper()

# 设定基础数据目录
DATA_DIR = os.path.join(project_root, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 根据环境决定数据库文件名
if SYSTEM_ENV == 'DEV':
    DB_NAME = 'trading_system_dev.db'
    LOG_LEVEL = logging.DEBUG
else:
    DB_NAME = 'trading_system_prod.db'
    LOG_LEVEL = logging.INFO

# 最终导出数据库路径
DB_PATH = os.path.join(DATA_DIR, DB_NAME)

# 其他全局配置变量
GOOGLE_APPLICATION_CREDENTIALS = os.path.join(project_root, 'vertex_key.json')
GCP_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'synthmind-social-content')
GCP_LOCATION = os.getenv('GCP_LOCATION', 'global')

PROXY_MODE = os.getenv('PROXY_MODE', 'direct').lower().strip()
PROXY_URL = os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY') or os.getenv('https_proxy') or os.getenv('http_proxy') or ''


def setup_proxy():
    proxy_keys = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
    for key in proxy_keys:
        os.environ.pop(key, None)
    os.environ.pop('no_proxy', None)
    os.environ.pop('NO_PROXY', None)

    if PROXY_MODE == 'proxy' and PROXY_URL:
        os.environ['http_proxy'] = PROXY_URL
        os.environ['https_proxy'] = PROXY_URL
        os.environ['HTTP_PROXY'] = PROXY_URL
        os.environ['HTTPS_PROXY'] = PROXY_URL
        logger.info(f"代理模式: PROXY ({PROXY_URL})")
    else:
        os.environ['no_proxy'] = '*'
        os.environ['NO_PROXY'] = '*'
        logger.info("代理模式: DIRECT (TUN直连)")


# 币安 API 配置
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")

from logger_setup import get_logger
logger = get_logger(__name__)

setup_proxy()

logger.info(f"配置加载完成 [环境: {SYSTEM_ENV}]")
logger.info(f"数据库路径: {DB_PATH}")
logger.info(f"日志级别: {logging.getLevelName(LOG_LEVEL)}")

if __name__ == "__main__":
    """
    测试配置模块
    """
    print("===== 配置测试 =====")
    print(f"SYSTEM_ENV: {SYSTEM_ENV}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"DB_PATH: {DB_PATH}")
    print(f"LOG_LEVEL: {logging.getLevelName(LOG_LEVEL)}")
    print(f"GCP_PROJECT_ID: {GCP_PROJECT_ID}")
    print(f"BINANCE_API_KEY: {'***已配置***' if BINANCE_API_KEY else '未配置'}")
    print("===== 配置测试完成 =====")
