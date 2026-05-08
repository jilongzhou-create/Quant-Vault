#!/usr/bin/env python3
"""
数据加载与清洗模块

从项目根目录下的 SQLite 数据库 (market_data_us_sectors 表) 读取 11 只 SPDR 行业 ETF 的日线数据，
返回形状对齐的字典 data_dict = {'open': df_O, 'high': df_H, ...}。

若数据库不可用，自动回退到 numpy 随机游走矩阵，保证可独立测试。
"""

import sys
import os
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

ETF_SYMBOLS = [
    'XLK', 'XLF', 'XLV', 'XLE', 'XLY',
    'XLI', 'XLC', 'XLU', 'XLP', 'XLRE', 'XLB',
]

_FIELDS = ['open', 'high', 'low', 'close', 'volume']

_DEFAULT_DB_PATH = os.path.join(project_root, 'data', 'trading_system_prod.db')


def load_sample_data(
    start_date: str = '2017-01-01',
    end_date: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    从 market_data_us_sectors 表加载 11 只 SPDR 行业 ETF 的日线 OHLCV 数据

    Args:
        start_date: 起始日期 (YYYY-MM-DD)
        end_date:   结束日期 (YYYY-MM-DD)，None 表示到最新
        db_path:    数据库文件路径，None 使用默认路径

    Returns:
        dict: {'open': df, 'high': df, 'low': df, 'close': df, 'volume': df}
              每个 df 的 shape = (T, N), index=Datetime, columns=ETF_SYMBOLS
    """
    try:
        return _load_from_db(start_date, end_date, db_path)
    except Exception as e:
        print(f"[DATA-LOADER] 数据库读取失败 ({e})，使用随机游走数据替代")
        return _generate_random_walk(start_date, end_date or '2026-05-01')


def _load_from_db(
    start_date: str,
    end_date: Optional[str],
    db_path: Optional[str],
) -> Dict[str, pd.DataFrame]:
    if db_path is None:
        db_path = _DEFAULT_DB_PATH

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")

    conn = sqlite3.connect(db_path)

    try:
        result: Dict[str, pd.DataFrame] = {}

        for field in _FIELDS:
            query = """
                SELECT date, symbol, {field}
                FROM market_data_us_sectors
                WHERE date >= ?
            """.format(field=field)
            params: list = [start_date]

            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date ASC, symbol ASC"

            df = pd.read_sql_query(query, conn, params=params)

            if df.empty:
                raise ValueError(f"market_data_us_sectors 表无数据 (field={field})")

            wide_df = df.pivot(index='date', columns='symbol', values=field)

            for symbol in ETF_SYMBOLS:
                if symbol not in wide_df.columns:
                    wide_df[symbol] = np.nan

            wide_df = wide_df.reindex(columns=ETF_SYMBOLS)
            wide_df.index = pd.to_datetime(wide_df.index)
            wide_df = wide_df.sort_index()

            result[field] = wide_df

    finally:
        conn.close()

    _validate_alignment(result)

    close_df = result['close']
    print(
        f"[DATA-LOADER] 数据库加载成功, "
        f"日期范围: {close_df.index[0].date()} ~ {close_df.index[-1].date()}, "
        f"shape={close_df.shape}"
    )
    return result


def _generate_random_walk(
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    n_assets = len(ETF_SYMBOLS)

    np.random.seed(42)
    close = 100.0 * np.exp(np.cumsum(np.random.randn(n_days, n_assets) * 0.015, axis=0))
    close_df = pd.DataFrame(close, index=dates, columns=ETF_SYMBOLS)

    result: Dict[str, pd.DataFrame] = {}
    result['close'] = close_df
    result['open'] = close_df * (1 + np.random.randn(n_days, n_assets) * 0.003)
    result['high'] = close_df * (1 + np.abs(np.random.randn(n_days, n_assets)) * 0.008)
    result['low'] = close_df * (1 - np.abs(np.random.randn(n_days, n_assets)) * 0.008)
    result['volume'] = pd.DataFrame(
        np.random.randint(1_000_000, 50_000_000, size=(n_days, n_assets)),
        index=dates, columns=ETF_SYMBOLS, dtype=float,
    )

    _validate_alignment(result)
    print(f"[DATA-LOADER] 随机游走数据生成成功, shape={result['close'].shape}")
    return result


def _validate_alignment(data_dict: Dict[str, pd.DataFrame]) -> None:
    shapes = {k: df.shape for k, df in data_dict.items()}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"数据形状不一致: {shapes}")
