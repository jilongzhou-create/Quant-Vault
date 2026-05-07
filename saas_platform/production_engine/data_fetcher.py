"""
云端全域数据拉取器 (Cloud Data Fetcher)

职责：增量拉取最新行情量价 + 宏观/微观因子数据，全量 UPSERT 到 Supabase。
严格约束：
  1. 禁止全量拉取，仅拉取最近 1~2 年数据（252 交易日 + 充足预热）
  2. 绝对禁止写入本地 SQLite，数据落库终点必须是 Supabase
  3. 复用现有 API 请求逻辑，但下游终点重定向至 Supabase
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from saas_platform.saas_config import (
    FRED_API_KEY,
    FMP_API_KEY,
    COINMETRICS_API_KEY,
    PROXY_MODE,
    PROXY_URL,
)
from saas_platform.database.supabase_client import (
    upsert_market_data,
    upsert_factor_data,
    get_latest_market_timestamp,
    get_client,
)

logger = logging.getLogger('saas_platform.production_engine.data_fetcher')

CLOUD_LOOKBACK_DAYS = 400


def _create_session() -> requests.Session:
    session = requests.Session()
    proxies = {}
    if PROXY_MODE == 'proxy' and PROXY_URL:
        proxies['http'] = PROXY_URL
        proxies['https'] = PROXY_URL
    if proxies:
        session.proxies.update(proxies)
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    return session


def _calc_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).fillna(0)


def _calc_macd(close: pd.Series):
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd.fillna(0), macd_signal.fillna(0), macd_hist.fillna(0)


# ============================================================
# 加密货币行情 (Binance REST API 直连，不依赖 ccxt)
# ============================================================

BINANCE_INTERVAL_MAP = {
    '1d': '1d',
    '4h': '4h',
    '1h': '1h',
}


def fetch_crypto_market(symbol: str = 'BTC/USDT', timeframe: str = '1d') -> int:
    """
    从 Binance REST API 拉取加密货币日线行情，仅拉取最近 CLOUD_LOOKBACK_DAYS 天
    不依赖 ccxt，直接用 requests 调用 /api/v3/klines，更轻量更稳定
    数据落库到 Supabase saas_market_data
    """
    db_symbol = symbol.replace('/', '_')
    binance_symbol = symbol.replace('/', '')
    interval = BINANCE_INTERVAL_MAP.get(timeframe, '1d')
    logger.info(f"[Crypto] 开始拉取 {symbol} 行情数据 (Binance REST API, interval={interval})...")

    try:
        session = _create_session()
        since_ms = int((datetime.now(timezone.utc) - timedelta(days=CLOUD_LOOKBACK_DAYS)).timestamp() * 1000)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        all_ohlcv = []
        current_start = since_ms

        while current_start < now_ms:
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'startTime': current_start,
                'limit': 1000,
            }
            response = session.get(
                'https://api.binance.com/api/v3/klines',
                params=params, timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            for item in data:
                all_ohlcv.append({
                    'timestamp': int(item[0]),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[5]),
                })

            last_ts = int(data[-1][0])
            if last_ts <= current_start:
                break
            current_start = last_ts + 1
            time.sleep(0.3)

        if not all_ohlcv:
            logger.warning(f"[Crypto] {symbol} 无数据")
            return 0

        df = pd.DataFrame(all_ohlcv)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['rsi_14'] = _calc_rsi(df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = _calc_macd(df['close'])
        df = df.dropna()
        df['symbol'] = db_symbol

        records = []
        for _, row in df.iterrows():
            records.append({
                'symbol': db_symbol,
                'timestamp': row['timestamp'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'rsi_14': float(row['rsi_14']) if not np.isnan(row['rsi_14']) else None,
                'macd': float(row['macd']),
                'macd_signal': float(row['macd_signal']),
                'macd_hist': float(row['macd_hist']),
            })

        count = upsert_market_data(records)
        logger.info(f"[Crypto] {symbol} 行情推送完成: {count} 条")
        return count

    except Exception as e:
        logger.error(f"[Crypto] {symbol} 行情拉取失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0


# ============================================================
# 美股/大宗商品行情 (FMP)
# ============================================================

def fetch_fmp_market(symbol: str, asset_type: str = 'us_stock') -> int:
    """
    从 FMP 拉取美股或大宗商品日线行情
    数据落库到 Supabase saas_market_data
    """
    if not FMP_API_KEY:
        logger.error("[FMP] FMP_API_KEY 未配置")
        return 0

    db_symbol = symbol
    logger.info(f"[FMP] 开始拉取 {symbol} ({asset_type}) 行情数据...")

    try:
        session = _create_session()
        from_date = (datetime.now(timezone.utc) - timedelta(days=CLOUD_LOOKBACK_DAYS)).strftime('%Y-%m-%d')
        to_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={symbol}&from={from_date}&to={to_date}&apikey={FMP_API_KEY}"
        response = session.get(url, timeout=30)

        if response.status_code != 200:
            logger.error(f"[FMP] {symbol} HTTP {response.status_code}")
            return 0

        data = response.json()
        if not isinstance(data, list) or len(data) == 0:
            logger.warning(f"[FMP] {symbol} 无数据")
            return 0

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['date'])
        df['symbol'] = db_symbol
        df['rsi_14'] = _calc_rsi(df['close'].astype(float))
        df['macd'], df['macd_signal'], df['macd_hist'] = _calc_macd(df['close'].astype(float))

        records = []
        for _, row in df.iterrows():
            records.append({
                'symbol': db_symbol,
                'timestamp': row['timestamp'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0) or 0),
                'rsi_14': float(row['rsi_14']) if not np.isnan(row['rsi_14']) else None,
                'macd': float(row['macd']),
                'macd_signal': float(row['macd_signal']),
                'macd_hist': float(row['macd_hist']),
            })

        count = upsert_market_data(records)
        logger.info(f"[FMP] {symbol} 行情推送完成: {count} 条")
        return count

    except Exception as e:
        logger.error(f"[FMP] {symbol} 行情拉取失败: {e}")
        return 0


# ============================================================
# FRED 宏观因子
# ============================================================

FRED_CORE_SERIES = [
    "M2SL", "WALCL", "FEDFUNDS", "CPIAUCSL", "UNRATE",
    "T10Y2Y", "DGS10", "BAMLH0A0HYM2", "VIXCLS", "WTISPLC",
    "STLFSI4", "NFCI",
]

FRED_LAG_MAP = {
    'daily': 1,
    'weekly': 7,
    'monthly': 35,
    'quarterly': 45,
}


def _infer_lag_days(median_gap_days: float) -> int:
    if median_gap_days <= 3:
        return 1
    elif median_gap_days <= 10:
        return 7
    elif median_gap_days <= 40:
        return 35
    else:
        return 45


def fetch_fred_factors() -> int:
    """
    从 FRED 增量拉取核心宏观因子数据
    数据落库到 Supabase saas_factor_data
    """
    if not FRED_API_KEY:
        logger.error("[FRED] FRED_API_KEY 未配置")
        return 0

    logger.info(f"[FRED] 开始拉取 {len(FRED_CORE_SERIES)} 个核心宏观因子...")
    session = _create_session()
    total = 0

    for idx, series_id in enumerate(FRED_CORE_SERIES, 1):
        factor_name = series_id.lower()
        logger.info(f"[FRED] [{idx}/{len(FRED_CORE_SERIES)}] {series_id}")

        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "observation_start": (datetime.now(timezone.utc) - timedelta(days=CLOUD_LOOKBACK_DAYS)).strftime('%Y-%m-%d'),
            }
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            time.sleep(0.6)

            data = response.json()
            observations = data.get('observations', [])

            if not observations:
                logger.warning(f"[FRED] {series_id} 无数据")
                continue

            df = pd.DataFrame(observations)
            df['date_parsed'] = pd.to_datetime(df['date'])
            df = df[df['value'] != '.']

            if df.empty:
                continue

            if len(df) > 1:
                median_gap = df['date_parsed'].diff().median().days
            else:
                median_gap = 30
            lag_days = _infer_lag_days(median_gap)

            df['event_timestamp'] = df['date_parsed'] + pd.Timedelta(days=lag_days)

            records = []
            for _, row in df.iterrows():
                records.append({
                    'symbol': 'MACRO',
                    'timestamp': row['event_timestamp'].strftime('%Y-%m-%dT23:59:59'),
                    'factor_name': factor_name,
                    'factor_value': float(row['value']),
                })

            if records:
                count = upsert_factor_data(records)
                total += count
                logger.info(f"[FRED] {series_id} 推送完成: {count} 条")

        except Exception as e:
            logger.error(f"[FRED] {series_id} 拉取失败: {e}")
            continue

    logger.info(f"[FRED] 宏观因子拉取完成，共 {total} 条")
    return total


# ============================================================
# CoinMetrics 链上因子
# ============================================================

COINMETRICS_METRICS = {
    'CapMVRVCur': 'mvrv',
    'SplyCur': 'supply_current',
    'AdrActCnt': 'active_addresses',
}


def fetch_coinmetrics_factors() -> int:
    """
    从 CoinMetrics 拉取链上因子数据
    数据落库到 Supabase saas_factor_data
    """
    logger.info("[CoinMetrics] 开始拉取链上因子数据...")
    session = _create_session()
    total = 0

    for metric, factor_name in COINMETRICS_METRICS.items():
        try:
            url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
            params = {
                "assets": "btc",
                "metrics": metric,
                "frequency": "1d",
                "start_time": (datetime.now(timezone.utc) - timedelta(days=CLOUD_LOOKBACK_DAYS)).strftime('%Y-%m-%d'),
            }
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            time.sleep(0.5)

            data = response.json().get('data', [])
            if not data:
                logger.warning(f"[CoinMetrics] {metric} 无数据")
                continue

            records = []
            for item in data:
                value_str = item.get(metric)
                if value_str and value_str != 'None':
                    try:
                        records.append({
                            'symbol': 'BTC_USDT',
                            'timestamp': pd.to_datetime(item['time']).strftime('%Y-%m-%dT23:59:59'),
                            'factor_name': factor_name,
                            'factor_value': float(value_str),
                        })
                    except (ValueError, TypeError):
                        continue

            if records:
                count = upsert_factor_data(records)
                total += count
                logger.info(f"[CoinMetrics] {metric} -> {factor_name} 推送完成: {count} 条")

        except Exception as e:
            logger.error(f"[CoinMetrics] {metric} 拉取失败: {e}")
            continue

    logger.info(f"[CoinMetrics] 链上因子拉取完成，共 {total} 条")
    return total


# ============================================================
# Binance 资金费率因子
# ============================================================

def fetch_funding_rate() -> int:
    """
    从 Binance Futures 拉取资金费率因子
    数据落库到 Supabase saas_factor_data
    """
    logger.info("[Binance] 开始拉取资金费率因子...")
    session = _create_session()
    total = 0

    try:
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        start_ts = int((datetime.now(timezone.utc) - timedelta(days=CLOUD_LOOKBACK_DAYS)).timestamp() * 1000)
        end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

        all_data = []
        current_end = end_ts

        while current_end > start_ts:
            params = {
                "symbol": "BTCUSDT",
                "startTime": start_ts,
                "endTime": current_end,
                "limit": 1000,
            }
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_data.extend(data)
            earliest = min(int(d['fundingTime']) for d in data)
            current_end = earliest - 1
            time.sleep(0.3)

        if not all_data:
            logger.warning("[Binance] 资金费率无数据")
            return 0

        records = []
        for item in all_data:
            records.append({
                'symbol': 'BTC_USDT',
                'timestamp': pd.to_datetime(int(item['fundingTime']), unit='ms').strftime('%Y-%m-%dT%H:%M:%S'),
                'factor_name': 'funding_rate',
                'factor_value': float(item['fundingRate']),
            })

        if records:
            total = upsert_factor_data(records)
            logger.info(f"[Binance] 资金费率推送完成: {total} 条")

    except Exception as e:
        logger.error(f"[Binance] 资金费率拉取失败: {e}")

    return total


# ============================================================
# 恐惧贪婪指数因子
# ============================================================

def fetch_fear_greed() -> int:
    """
    从 Alternative.me 拉取恐惧与贪婪指数
    数据落库到 Supabase saas_factor_data
    """
    logger.info("[FearGreed] 开始拉取恐惧贪婪指数...")
    session = _create_session()

    try:
        url = "https://api.alternative.me/fng/"
        params = {"limit": 0}
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json().get('data', [])
        if not data:
            logger.warning("[FearGreed] 无数据")
            return 0

        cutoff = (datetime.now(timezone.utc) - timedelta(days=CLOUD_LOOKBACK_DAYS))

        records = []
        for item in data:
            ts = pd.to_datetime(int(item['timestamp']), unit='s')
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
            if ts < cutoff:
                continue
            records.append({
                'symbol': 'BTC_USDT',
                'timestamp': ts.strftime('%Y-%m-%dT23:59:59'),
                'factor_name': 'fear_greed',
                'factor_value': float(item['value']),
            })

        if records:
            count = upsert_factor_data(records)
            logger.info(f"[FearGreed] 推送完成: {count} 条")
            return count

    except Exception as e:
        logger.error(f"[FearGreed] 拉取失败: {e}")

    return 0


# ============================================================
# 全域同步主入口
# ============================================================

SYMBOL_CONFIG = {
    'crypto': [
        {'symbol': 'BTC/USDT', 'db_symbol': 'BTC_USDT', 'fetcher': 'crypto'},
    ],
    'us_stock': [
        {'symbol': 'SPY', 'db_symbol': 'SPY', 'fetcher': 'fmp'},
        {'symbol': 'QQQ', 'db_symbol': 'QQQ', 'fetcher': 'fmp'},
    ],
    'commodity': [
        {'symbol': 'GCUSD', 'db_symbol': 'GCUSD', 'fetcher': 'fmp'},
        {'symbol': 'BZUSD', 'db_symbol': 'BZUSD', 'fetcher': 'fmp'},
    ],
}


def sync_all_market_data() -> dict:
    """
    同步所有标的的行情数据到 Supabase
    """
    results = {}
    logger.info("=" * 60)
    logger.info("开始全域行情数据同步")
    logger.info("=" * 60)

    for asset_type, symbols in SYMBOL_CONFIG.items():
        for cfg in symbols:
            key = cfg['db_symbol']
            if cfg['fetcher'] == 'crypto':
                count = fetch_crypto_market(symbol=cfg['symbol'])
            else:
                count = fetch_fmp_market(symbol=cfg['symbol'], asset_type=asset_type)
            results[key] = count

    logger.info(f"行情同步完成: {results}")
    return results


def sync_all_factor_data() -> dict:
    """
    同步所有因子数据到 Supabase
    """
    results = {}
    logger.info("=" * 60)
    logger.info("开始全域因子数据同步")
    logger.info("=" * 60)

    results['fred'] = fetch_fred_factors()
    results['coinmetrics'] = fetch_coinmetrics_factors()
    results['funding_rate'] = fetch_funding_rate()
    results['fear_greed'] = fetch_fear_greed()

    logger.info(f"因子同步完成: {results}")
    return results


def run_daily_sync() -> dict:
    """
    每日定时任务入口：先同步行情，再同步因子
    """
    logger.info("=" * 60)
    logger.info("🚀 每日全域数据同步开始")
    logger.info("=" * 60)

    market_results = sync_all_market_data()
    factor_results = sync_all_factor_data()

    total_market = sum(market_results.values())
    total_factors = sum(factor_results.values())

    logger.info("=" * 60)
    logger.info(f"✅ 每日同步完成: 行情 {total_market} 条, 因子 {total_factors} 条")
    logger.info("=" * 60)

    return {
        'market': market_results,
        'factors': factor_results,
        'total_market': total_market,
        'total_factors': total_factors,
    }
