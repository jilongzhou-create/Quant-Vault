"""
SaaS 平台 Supabase 数据库客户端 (REST API 直连版)

通过 PostgREST API 直接操作 Supabase，无需安装 supabase-py。
仅依赖 requests 库，兼容所有 Python 版本。

单例模式，通过 saas_config 获取独立配置，与本地投研系统完全物理隔离。
"""

import logging
from typing import Optional

import requests

from saas_platform.saas_config import SAAS_SUPABASE_URL, SAAS_SUPABASE_KEY, is_configured, PROXY_MODE, PROXY_URL

logger = logging.getLogger('saas_platform.database')


class SupabaseClient:
    _instance: Optional['SupabaseClient'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        if not is_configured():
            raise RuntimeError("SaaS Supabase 未配置，请设置 SAAS_SUPABASE_URL 和 SAAS_SUPABASE_KEY")
        self._base_url = SAAS_SUPABASE_URL.rstrip('/')
        if self._base_url.endswith('/rest/v1'):
            self._rest_url = self._base_url
        else:
            self._rest_url = f"{self._base_url}/rest/v1"
        self._headers = {
            'apikey': SAAS_SUPABASE_KEY,
            'Authorization': f'Bearer {SAAS_SUPABASE_KEY}',
            'Content-Type': 'application/json',
            'Prefer': 'return=representation',
        }
        if PROXY_MODE == 'proxy' and PROXY_URL:
            self._proxies = {'http': PROXY_URL, 'https': PROXY_URL}
        else:
            self._proxies = None
        self._session = requests.Session()
        if self._proxies:
            self._session.proxies.update(self._proxies)
        self._initialized = True
        logger.info(f"Supabase REST 客户端初始化成功 (proxy={PROXY_MODE})")

    def _request(self, method: str, table: str, *,
                 json_data=None, params: dict = None) -> list[dict]:
        url = f"{self._rest_url}/{table}"
        try:
            resp = self._session.request(
                method, url,
                headers=self._headers,
                json=json_data,
                params=params,
                timeout=30,
            )
        except requests.exceptions.SSLError:
            logger.warning("SSL 错误，尝试跳过代理直连...")
            resp = requests.request(
                method, url,
                headers=self._headers,
                json=json_data,
                params=params,
                timeout=30,
            )
        if resp.status_code >= 400:
            logger.error(f"Supabase {method} {table} 失败: {resp.status_code} {resp.text[:500]}")
            resp.raise_for_status()
        if resp.text.strip():
            return resp.json()
        return []

    def select(self, table: str, columns: str = '*',
               filters: dict = None, order: str = None,
               limit: int = None, offset: int = None) -> list[dict]:
        params = {'select': columns}
        if filters:
            params.update(filters)
        if order:
            params['order'] = order
        if limit:
            params['limit'] = limit
        if offset:
            params['offset'] = offset
        return self._request('GET', table, params=params)

    def insert(self, table: str, records) -> list[dict]:
        if isinstance(records, dict):
            records = [records]
        return self._request('POST', table, json_data=records)

    def upsert(self, records, table: str, on_conflict: str = None) -> list[dict]:
        if isinstance(records, dict):
            records = [records]
        headers = {**self._headers}
        prefer = 'return=representation,resolution=merge-duplicates'
        headers['Prefer'] = prefer
        params = {}
        if on_conflict:
            params['on_conflict'] = on_conflict
        url = f"{self._rest_url}/{table}"
        try:
            resp = self._session.request(
                'POST', url,
                headers=headers,
                json=records,
                params=params,
                timeout=30,
            )
        except requests.exceptions.SSLError:
            logger.warning("SSL 错误(upsert)，尝试跳过代理直连...")
            resp = requests.request(
                'POST', url,
                headers=headers,
                json=records,
                params=params,
                timeout=30,
            )
        if resp.status_code >= 400:
            logger.error(f"Supabase upsert {table} 失败: {resp.status_code} {resp.text[:500]}")
            resp.raise_for_status()
        if resp.text.strip():
            return resp.json()
        return []

    def update(self, table: str, filters: dict, data: dict) -> list[dict]:
        headers = {**self._headers, 'Prefer': 'return=representation'}
        url = f"{self._rest_url}/{table}"
        try:
            resp = self._session.request(
                'PATCH', url,
                headers=headers,
                json=data,
                params=filters,
                timeout=30,
            )
        except requests.exceptions.SSLError:
            logger.warning("SSL 错误(update)，尝试跳过代理直连...")
            resp = requests.request(
                'PATCH', url,
                headers=headers,
                json=data,
                params=filters,
                timeout=30,
            )
        if resp.status_code >= 400:
            logger.error(f"Supabase update {table} 失败: {resp.status_code} {resp.text[:500]}")
            resp.raise_for_status()
        if resp.text.strip():
            return resp.json()
        return []

    def delete(self, table: str, filters: dict) -> list[dict]:
        headers = {**self._headers, 'Prefer': 'return=representation'}
        url = f"{self._rest_url}/{table}"
        try:
            resp = self._session.request(
                'DELETE', url,
                headers=headers,
                params=filters,
                timeout=30,
            )
        except requests.exceptions.SSLError:
            logger.warning("SSL 错误(delete)，尝试跳过代理直连...")
            resp = requests.request(
                'DELETE', url,
                headers=headers,
                params=filters,
                timeout=30,
            )
        if resp.status_code >= 400:
            logger.error(f"Supabase delete {table} 失败: {resp.status_code} {resp.text[:500]}")
            resp.raise_for_status()
        if resp.text.strip():
            return resp.json()
        return []

    def rpc(self, function_name: str, params: dict = None) -> list[dict]:
        url = f"{self._rest_url}/rpc/{function_name}"
        try:
            resp = self._session.request(
                'POST', url,
                headers=self._headers,
                json=params or {},
                timeout=30,
            )
        except requests.exceptions.SSLError:
            logger.warning("SSL 错误(rpc)，尝试跳过代理直连...")
            resp = requests.request(
                'POST', url,
                headers=self._headers,
                json=params or {},
                timeout=30,
            )
        if resp.status_code >= 400:
            logger.error(f"Supabase rpc {function_name} 失败: {resp.status_code} {resp.text[:500]}")
            resp.raise_for_status()
        if resp.text.strip():
            return resp.json()
        return []

    @classmethod
    def reset(cls):
        cls._instance = None


def get_client() -> Optional[SupabaseClient]:
    try:
        return SupabaseClient()
    except RuntimeError as e:
        logger.warning(f"Supabase 客户端不可用: {e}")
        return None


# ============================================================
# 行情数据接口
# ============================================================

def upsert_market_data(records: list[dict]) -> int:
    if not records:
        return 0
    db = get_client()
    if not db:
        return 0
    total = 0
    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        result = db.upsert(batch, 'saas_market_data', on_conflict='symbol,timestamp')
        count = len(result) if result else 0
        total += count
    logger.info(f"upsert_market_data: {total} 条 (分 {(len(records) + batch_size - 1) // batch_size} 批)")
    return total


def get_market_data(symbol: str, limit: int = 252, order: str = 'timestamp.asc') -> list[dict]:
    db = get_client()
    if not db:
        return []
    all_records = []
    offset = 0
    page_size = 1000
    while True:
        batch = db.select(
            'saas_market_data', columns='*',
            filters={'symbol': f'eq.{symbol}'},
            order=order, limit=page_size, offset=offset,
        )
        if not batch:
            break
        all_records.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
        if limit < 99999 and len(all_records) >= limit:
            break
    if limit < 99999:
        return all_records[:limit]
    return all_records


def get_latest_market_timestamp(symbol: str) -> Optional[str]:
    db = get_client()
    if not db:
        return None
    result = db.select(
        'saas_market_data', columns='timestamp',
        filters={'symbol': f'eq.{symbol}'},
        order='timestamp.desc', limit=1,
    )
    if result:
        return result[0]['timestamp']
    return None


# ============================================================
# 因子数据接口
# ============================================================

def upsert_factor_data(records: list[dict]) -> int:
    if not records:
        return 0
    db = get_client()
    if not db:
        return 0
    total = 0
    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        result = db.upsert(batch, 'saas_factor_data', on_conflict='symbol,timestamp,factor_name')
        count = len(result) if result else 0
        total += count
    logger.info(f"upsert_factor_data: {total} 条 (分 {(len(records) + batch_size - 1) // batch_size} 批)")
    return total


def get_factor_data(symbol: str, factor_names: list[str] = None, limit: int = 5000, order: str = 'timestamp.asc') -> list[dict]:
    db = get_client()
    if not db:
        return []
    filters = {'symbol': f'eq.{symbol}'}
    if factor_names:
        filters['factor_name'] = f'in.({",".join(factor_names)})'
    effective_limit = limit * max(len(factor_names), 1) if factor_names else limit
    page_size = 1000
    all_records = []
    offset = 0
    while True:
        batch = db.select(
            'saas_factor_data', columns='*',
            filters=filters,
            order=order,
            limit=page_size, offset=offset,
        )
        if not batch:
            break
        all_records.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
        if effective_limit < 99999 and len(all_records) >= effective_limit:
            break
    if effective_limit < 99999:
        return all_records[:effective_limit]
    return all_records


def upsert_factor_metadata(records: list[dict]) -> int:
    if not records:
        return 0
    db = get_client()
    if not db:
        return 0
    result = db.upsert(records, 'saas_factor_metadata', on_conflict='factor_name,symbol')
    count = len(result) if result else 0
    logger.info(f"upsert_factor_metadata: {count} 条")
    return count


# ============================================================
# 策略接口
# ============================================================

def publish_strategy(strategy_data: dict) -> dict:
    db = get_client()
    if not db:
        return {}
    result = db.insert('saas_strategies', strategy_data)
    if result:
        logger.info(f"策略已发布: {strategy_data.get('name', 'unknown')}")
        return result[0]
    return {}


def upsert_strategy(strategy_data: dict) -> dict:
    if not strategy_data:
        return {}
    db = get_client()
    if not db:
        return {}
    on_conflict = 'id' if 'id' in strategy_data else 'name'
    result = db.upsert(strategy_data, 'saas_strategies', on_conflict=on_conflict)
    if result:
        logger.info(f"策略已 upsert: {strategy_data.get('name', 'unknown')}")
        return result[0]
    return {}


def get_strategy_code(strategy_id: str) -> Optional[dict]:
    db = get_client()
    if not db:
        return None
    result = db.select(
        'saas_strategies',
        columns='python_code,params_json,target_symbol,target_asset,timeframe',
        filters={'id': f'eq.{strategy_id}', 'status': 'eq.LIVE'},
    )
    if result:
        return result[0]
    return None


def get_live_strategies(target_symbol: str = None) -> list[dict]:
    db = get_client()
    if not db:
        return []
    filters = {'status': 'in.(LIVE,PAPER)'}
    if target_symbol:
        filters['target_symbol'] = f'eq.{target_symbol}'
    return db.select(
        'saas_strategies',
        columns='id,name,target_symbol,target_asset,current_target_position,backtest_sharpe,status',
        filters=filters,
    )


def update_strategy_position(strategy_id: str, position: float) -> bool:
    db = get_client()
    if not db:
        return False
    result = db.update(
        'saas_strategies',
        filters={'id': f'eq.{strategy_id}'},
        data={'current_target_position': position},
    )
    return bool(result)


# ============================================================
# 净值曲线接口
# ============================================================

def upsert_equity_curve(records: list[dict]) -> int:
    if not records:
        return 0
    db = get_client()
    if not db:
        return 0
    result = db.upsert(records, 'saas_equity_curves', on_conflict='strategy_id,date,is_backtest')
    count = len(result) if result else 0
    logger.info(f"upsert_equity_curve: {count} 条")
    return count


def bulk_upsert_equity_curves(curves_data_list: list[dict], batch_size: int = 500) -> int:
    if not curves_data_list:
        return 0
    db = get_client()
    if not db:
        return 0
    total = 0
    for i in range(0, len(curves_data_list), batch_size):
        batch = curves_data_list[i:i + batch_size]
        result = db.upsert(batch, 'saas_equity_curves', on_conflict='strategy_id,date,is_backtest')
        count = len(result) if result else 0
        total += count
        logger.info(f"bulk_upsert_equity_curves: 批次 {i // batch_size + 1}, {count} 条")
    logger.info(f"bulk_upsert_equity_curves 完成: 共 {total} 条")
    return total


def update_daily_nav(strategy_id: str, date: str, nav_value: float, is_backtest: bool = False) -> bool:
    db = get_client()
    if not db:
        return False
    result = db.upsert(
        {
            'strategy_id': strategy_id,
            'date': date,
            'nav_value': nav_value,
            'is_backtest': is_backtest,
        },
        'saas_equity_curves',
        on_conflict='strategy_id,date,is_backtest',
    )
    return bool(result)


def delete_equity_curves(strategy_id: str, is_backtest: bool = None) -> int:
    db = get_client()
    if not db:
        return 0
    filters = {'strategy_id': f'eq.{strategy_id}'}
    if is_backtest is not None:
        filters['is_backtest'] = f'eq.{str(is_backtest).lower()}'
    result = db.delete('saas_equity_curves', filters=filters)
    count = len(result) if result else 0
    logger.info(f"delete_equity_curves: {count} 条 (strategy_id={strategy_id[:8]}..., is_backtest={is_backtest})")
    return count


def get_equity_curve(strategy_id: str, is_backtest: bool = None, limit: int = 365) -> list[dict]:
    db = get_client()
    if not db:
        return []
    filters = {'strategy_id': f'eq.{strategy_id}'}
    if is_backtest is not None:
        filters['is_backtest'] = f'eq.{str(is_backtest).lower()}'
    all_records = []
    offset = 0
    page_size = 1000
    while True:
        batch = db.select(
            'saas_equity_curves',
            columns='date,nav_value,is_backtest',
            filters=filters,
            order='date.desc', limit=page_size, offset=offset,
        )
        if not batch:
            break
        all_records.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
        if limit < 99999 and len(all_records) >= limit:
            break
    if limit < 99999:
        return all_records[:limit]
    return all_records


# ============================================================
# 用户与订阅接口
# ============================================================

def create_user(user_data: dict) -> dict:
    db = get_client()
    if not db:
        return {}
    result = db.insert('saas_users', user_data)
    return result[0] if result else {}


def get_user_by_auth_id(auth_user_id: str) -> Optional[dict]:
    db = get_client()
    if not db:
        return None
    result = db.select(
        'saas_users', columns='*',
        filters={'auth_user_id': f'eq.{auth_user_id}'},
    )
    return result[0] if result else None


def get_user_by_id(user_id: str) -> Optional[dict]:
    db = get_client()
    if not db:
        return None
    result = db.select(
        'saas_users', columns='*',
        filters={'id': f'eq.{user_id}'},
    )
    return result[0] if result else None


def create_subscription(user_id: str, strategy_id: str, allocated_capital_usdt: float) -> dict:
    db = get_client()
    if not db:
        return {}
    result = db.insert('saas_subscriptions', {
        'user_id': user_id,
        'strategy_id': strategy_id,
        'allocated_capital_usdt': allocated_capital_usdt,
    })
    return result[0] if result else {}


def get_active_subscriptions(strategy_id: str = None) -> list[dict]:
    db = get_client()
    if not db:
        return []
    filters = {'is_active': 'eq.true'}
    if strategy_id:
        filters['strategy_id'] = f'eq.{strategy_id}'
    return db.select(
        'saas_subscriptions',
        columns='*,saas_users(username,exchange,encrypted_api_key,encrypted_api_secret)',
        filters=filters,
    )


def deactivate_subscription(subscription_id: int) -> bool:
    db = get_client()
    if not db:
        return False
    result = db.update(
        'saas_subscriptions',
        filters={'id': f'eq.{subscription_id}'},
        data={'is_active': False},
    )
    return bool(result)


# ============================================================
# 订单接口
# ============================================================

def create_order(order_data: dict) -> dict:
    db = get_client()
    if not db:
        return {}
    result = db.insert('saas_orders', order_data)
    return result[0] if result else {}


def update_order_status(order_id: int, status: str, exchange_order_id: str = '', error_message: str = '') -> bool:
    db = get_client()
    if not db:
        return False
    update_data = {'status': status}
    if exchange_order_id:
        update_data['exchange_order_id'] = exchange_order_id
    if error_message:
        update_data['error_message'] = error_message
    result = db.update(
        'saas_orders',
        filters={'id': f'eq.{order_id}'},
        data=update_data,
    )
    return bool(result)


# ============================================================
# AI 洞察接口
# ============================================================

def upsert_daily_insight(strategy_id: str, date: str, ai_analysis_text: str) -> bool:
    db = get_client()
    if not db:
        return False
    result = db.upsert(
        {
            'strategy_id': strategy_id,
            'date': date,
            'ai_analysis_text': ai_analysis_text,
        },
        'saas_daily_insights',
        on_conflict='strategy_id,date',
    )
    return bool(result)


def get_daily_insights(strategy_id: str, limit: int = 30) -> list[dict]:
    db = get_client()
    if not db:
        return []
    return db.select(
        'saas_daily_insights', columns='*',
        filters={'strategy_id': f'eq.{strategy_id}'},
        order='date.desc', limit=limit,
    )


# ============================================================
# 前端展示 & 跟单路由 补充接口
# ============================================================

def get_public_strategies() -> list[dict]:
    db = get_client()
    if not db:
        return []
    return db.select(
        'saas_strategies',
        columns='id,name,description,target_asset,target_symbol,current_target_position,backtest_sharpe,backtest_annualized_return,backtest_max_drawdown,backtest_start_date,backtest_end_date,live_start_date,status,timeframe',
        filters={'status': 'in.(LIVE,PAPER)'},
        order='backtest_sharpe.desc',
    )


def get_strategy_equity_curve(strategy_id: str, is_backtest: bool = None, limit: int = 10000) -> list[dict]:
    db = get_client()
    if not db:
        return []
    filters = {'strategy_id': f'eq.{strategy_id}'}
    if is_backtest is not None:
        filters['is_backtest'] = f'eq.{str(is_backtest).lower()}'
    all_records = []
    offset = 0
    page_size = 1000
    while True:
        batch = db.select(
            'saas_equity_curves',
            columns='date,nav_value,is_backtest',
            filters=filters,
            order='date.asc', limit=page_size, offset=offset,
        )
        if not batch:
            break
        all_records.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
        if limit < 99999 and len(all_records) >= limit:
            break
    if limit < 99999:
        return all_records[:limit]
    return all_records


def get_user_by_username(username: str) -> Optional[dict]:
    db = get_client()
    if not db:
        return None
    result = db.select(
        'saas_users', columns='*',
        filters={'username': f'eq.{username}'},
    )
    return result[0] if result else None


def update_user_api_keys(user_id: str, encrypted_api_key: str, encrypted_api_secret: str, exchange: str = 'binance') -> bool:
    db = get_client()
    if not db:
        return False
    result = db.update(
        'saas_users',
        filters={'id': f'eq.{user_id}'},
        data={
            'encrypted_api_key': encrypted_api_key,
            'encrypted_api_secret': encrypted_api_secret,
            'exchange': exchange,
        },
    )
    return bool(result)


def get_user_subscriptions(user_id: str) -> list[dict]:
    db = get_client()
    if not db:
        return []
    return db.select(
        'saas_subscriptions',
        columns='id,strategy_id,allocated_capital_usdt,is_active,subscribed_at,unsubscribed_at',
        filters={'user_id': f'eq.{user_id}'},
        order='subscribed_at.desc',
    )


def get_user_orders(user_id: str, limit: int = 50) -> list[dict]:
    db = get_client()
    if not db:
        return []
    return db.select(
        'saas_orders',
        columns='id,strategy_id,symbol,side,order_type,amount,price,fee,exchange_order_id,status,error_message,target_position,balance_before,balance_after,position_before,position_after,notional_value,is_sandbox,created_at',
        filters={'user_id': f'eq.{user_id}'},
        order='created_at.desc', limit=limit,
    )


def get_subscription_by_id(subscription_id: int) -> Optional[dict]:
    db = get_client()
    if not db:
        return None
    result = db.select(
        'saas_subscriptions',
        columns='id,user_id,strategy_id,allocated_capital_usdt,is_active',
        filters={'id': f'eq.{subscription_id}'},
    )
    return result[0] if result else None
