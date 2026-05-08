#!/usr/bin/env python3
"""
宏观估值核心模型 - Gold Macro Valuation Model

从 macro_model_registry 加载 β 系数，结合实时宏观数据，
计算每日理论价格 (Fair Value) 和估值偏差 (Valuation Spread)。

公式 (raw_level 模式):
  ln(FairValue) = intercept + b_DFII10 * DFII10 + b_lnDXY * ln(DTWEXBGS)
  FairValue = exp(ln(FairValue))
  ValuationSpread = ln(MarketPrice) - ln(FairValue)
                   = ln(MarketPrice / FairValue)
"""

import os
import sys
import json
import sqlite3

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DB_PATH


def load_model(model_id):
    """
    从 macro_model_registry 加载模型参数

    Returns:
        dict: {intercept, b_DFII10, b_lnDXY, data_mode, ...}
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT params_json, formula_desc, is_start_date, is_end_date FROM macro_model_registry WHERE model_id = ?",
        (model_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        raise ValueError(f"Model '{model_id}' not found in macro_model_registry")

    params = json.loads(row[0])
    params['formula_desc'] = row[1]
    params['is_start_date'] = row[2]
    params['is_end_date'] = row[3]
    params['model_id'] = model_id
    return params


def load_market_data(symbol, start_date=None, end_date=None):
    """加载黄金日线行情"""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT timestamp, close FROM market_data_gold WHERE symbol = ?"
    params = [symbol]
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    query += " ORDER BY timestamp"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize()
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
    df.rename(columns={'close': 'market_price'}, inplace=True)
    return df


def load_fred_raw_series(series_id, start_date=None, end_date=None):
    """从 raw_data 加载 FRED 原始水平值"""
    from database.db_manager import get_raw_data_by_source

    source_id = f'fred_{series_id}'
    raw_records = get_raw_data_by_source(source_id, start_time=pd.Timestamp(start_date) if start_date else None)

    rows = []
    for rec in raw_records:
        try:
            raw_content = rec.get('raw_content', rec) if isinstance(rec, dict) else json.loads(rec[4]) if len(rec) > 4 else {}
            if isinstance(raw_content, str):
                raw_content = json.loads(raw_content)
            value_str = raw_content.get('value', '.')
            event_ts = raw_content.get('date', rec.get('event_timestamp', ''))
            if value_str and value_str != '.':
                rows.append({'timestamp': event_ts, 'value': float(value_str)})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize()
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
    df.rename(columns={'value': series_id.lower()}, inplace=True)

    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    return df


def load_fred_factor_series(factor_name, start_date=None, end_date=None):
    """从 factor_data 加载 Z-Score 标准化值"""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT timestamp, factor_value FROM factor_data WHERE symbol='MACRO' AND factor_name = ?"
    params = [factor_name]
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    query += " ORDER BY timestamp"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize()
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
    df.rename(columns={'factor_value': factor_name}, inplace=True)
    return df


def load_macro_factor(factor_name, start_date=None, end_date=None):
    """
    从 factor_data 加载宏观因子原始值（通用版）

    与 load_fred_factor_series 不同，此函数:
      1. 不假设因子来自 FRED，适用于任何来源
      2. 列名使用传入的 factor_name 而非硬编码
      3. 支持 symbol='MACRO' 的标准宏观因子

    Args:
        factor_name: 因子名称 (e.g. 'bamlh0a0hym2', 'sge_au9999', 'sge_premium')
        start_date: 起始日期
        end_date: 结束日期

    Returns:
        DataFrame: index=timestamp, columns=[factor_name]
    """
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT timestamp, factor_value FROM factor_data WHERE symbol='MACRO' AND factor_name = ?"
    params = [factor_name]
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    query += " ORDER BY timestamp"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize()
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
    df.rename(columns={'factor_value': factor_name}, inplace=True)
    return df


def compute_fair_value(model_params, df_merged):
    """
    根据模型参数计算每日理论价格和估值偏差

    Args:
        model_params: 从 load_model() 返回的参数字典
        df_merged: 已合并的 DataFrame，包含 market_price, dfii10, dtwexbgs 列

    Returns:
        DataFrame: 新增 fair_value, valuation_spread 列
    """
    df = df_merged.copy()
    data_mode = model_params.get('data_mode', 'raw_level')
    intercept = model_params['intercept']
    b_dfii10 = model_params['b_DFII10']
    b_ln_dxy = model_params['b_lnDXY']

    if data_mode == 'raw_level':
        df['ln_dxy'] = np.log(df['dtwexbgs'])
        df['ln_fair'] = intercept + b_dfii10 * df['dfii10'] + b_ln_dxy * df['ln_dxy']
    else:
        df['ln_fair'] = intercept + b_dfii10 * df['dfii10'] + b_ln_dxy * df['dtwexbgs']

    df['fair_value'] = np.exp(df['ln_fair'])
    df['ln_market'] = np.log(df['market_price'])
    df['valuation_spread'] = df['ln_market'] - df['ln_fair']

    return df


def build_valuation_dataframe(model_id, start_date=None, end_date=None):
    """
    完整的估值计算管线：加载模型 → 加载数据 → 合并 → 计算理论价格与偏差

    Args:
        model_id: 模型 ID (e.g. 'gold_macro_v1')
        start_date: 起始日期 (str or None)
        end_date: 结束日期 (str or None)

    Returns:
        DataFrame: 包含 market_price, fair_value, valuation_spread 的完整估值表
    """
    model_params = load_model(model_id)
    data_mode = model_params.get('data_mode', 'raw_level')

    df_gold = load_market_data('GCUSD', start_date, end_date)

    if data_mode == 'raw_level':
        df_dfii10 = load_fred_raw_series('DFII10', start_date, end_date)
        df_dtwexbgs = load_fred_raw_series('DTWEXBGS', start_date, end_date)
    else:
        df_dfii10 = load_fred_factor_series('dfii10', start_date, end_date)
        df_dtwexbgs = load_fred_factor_series('dtwexbgs', start_date, end_date)

    df = df_gold.copy()
    if not df_dfii10.empty:
        df = df.join(df_dfii10[['dfii10']], how='left')
    if not df_dtwexbgs.empty:
        df = df.join(df_dtwexbgs[['dtwexbgs']], how='left')

    df = df.sort_index()
    df = df.ffill()
    df = df.dropna(subset=['market_price', 'dfii10', 'dtwexbgs'])

    df = compute_fair_value(model_params, df)

    return df, model_params
