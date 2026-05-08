#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Target Exposure Engine - 目标敞口引擎

基于宏观估值模型，计算每日估值偏差的 Z-Score，
并映射为目标敞口 target_exposure ∈ [-1.0, 1.0]。

核心逻辑:
  1. 从 core/valuation_model 获取每日估值偏差 (valuation_spread)
  2. 使用 IS 周期的偏差分布计算 Z-Score:
     z = (spread - mean_IS) / std_IS
  3. 将 Z-Score 映射为目标敞口:
     - z > 0: 市场价格高于理论价格 (超买) → 目标敞口 < 0 (减仓)
     - z < 0: 市场价格低于理论价格 (低估值) → 目标敞口 > 0 (加仓)
  4. 映射函数: target_exposure = -clip(z / z_cap, -1, 1)
  5. 结果落库到 macro_valuation_daily
"""

import io
import os
import sys
import json
import sqlite3

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DB_PATH
from isolated_macro.core.valuation_model import build_valuation_dataframe, load_model
from isolated_macro.core.gold_valuation import GoldMacroTrendV6, GoldMacroTrendV7, GoldMacroTrendV8

Z_SCORE_CAP = 2.0


def compute_is_distribution(model_id):
    """
    计算 IS 周期内 valuation_spread 的均值和标准差
    这些是"历史物理常数"，用于 OOS 期间的 Z-Score 标准化
    """
    model_params = load_model(model_id)
    is_start = model_params['is_start_date']
    is_end = model_params['is_end_date']

    df_is, _ = build_valuation_dataframe(model_id, start_date=is_start, end_date=is_end)

    if df_is.empty:
        raise ValueError(f"No data in IS period [{is_start}, {is_end}]")

    spread_mean = df_is['valuation_spread'].mean()
    spread_std = df_is['valuation_spread'].std()

    print(f"[IS Distribution] period: {is_start} ~ {is_end}, n={len(df_is)}")
    print(f"  spread_mean = {spread_mean:.6f}")
    print(f"  spread_std  = {spread_std:.6f}")

    return spread_mean, spread_std


def compute_spread_zscore(spread_series, spread_mean, spread_std):
    """计算偏差 Z-Score"""
    return (spread_series - spread_mean) / spread_std


def compute_target_exposure(z_score, z_cap=Z_SCORE_CAP):
    """
    将 Z-Score 映射为目标敞口

    逻辑:
      - z > 0: 价格高于理论 (超买) → 减仓 → exposure < 0
      - z < 0: 价格低于理论 (低估) → 加仓 → exposure > 0
      - 映射: exposure = -clip(z / z_cap, -1, 1)

    Args:
        z_score: Z-Score 值或 Series
        z_cap: Z-Score 饱和阈值，超过此值敞口封顶

    Returns:
        target_exposure: [-1.0, 1.0] 范围的目标敞口
    """
    normalized = np.clip(z_score / z_cap, -1.0, 1.0)
    return -normalized


def save_valuation_daily(df_valuation, model_id, symbol):
    """
    将估值结果批量保存到 macro_valuation_daily 表

    Args:
        df_valuation: DataFrame，需包含 timestamp(index), market_price, fair_value,
                      valuation_spread, spread_zscore, target_exposure
        model_id: 模型 ID
        symbol: 标的符号
    """
    if df_valuation.empty:
        print("[WARN] No data to save")
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    saved = 0
    for ts, row in df_valuation.iterrows():
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO macro_valuation_daily
                (timestamp, model_id, symbol, market_price, fair_value,
                 valuation_spread, spread_zscore, target_exposure)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ts.strftime('%Y-%m-%d'),
                model_id,
                symbol,
                float(row['market_price']),
                float(row['fair_value']),
                float(row['valuation_spread']),
                float(row['spread_zscore']),
                float(row['target_exposure']),
            ))
            saved += 1
        except Exception as e:
            print(f"[ERROR] Failed to save row {ts}: {e}")

    conn.commit()
    conn.close()
    print(f"[DB] Saved {saved} rows to macro_valuation_daily (model={model_id}, symbol={symbol})")
    return saved


def run_engine(model_id='gold_macro_v1', symbol='GCUSD',
               start_date=None, end_date=None):
    """
    运行 Target Exposure 引擎

    Args:
        model_id: 模型 ID ('gold_macro_v1' 或 'gold_macro_v6')
        symbol: 标的符号
        start_date: 起始日期 (None=全量)
        end_date: 结束日期 (None=全量)

    Returns:
        DataFrame: 完整估值结果
    """
    if model_id == 'gold_macro_v6':
        return run_engine_v6(symbol=symbol, start_date=start_date, end_date=end_date)
    if model_id == 'gold_macro_v7':
        return run_engine_v7(symbol=symbol, start_date=start_date, end_date=end_date)
    if model_id == 'gold_macro_v8':
        return run_engine_v8(symbol=symbol, start_date=start_date, end_date=end_date)

    print("=" * 78)
    print(f"Target Exposure Engine - {model_id} / {symbol}")
    print("=" * 78)

    # Step 1: 计算 IS 分布参数
    print("\n[Step 1] Computing IS distribution parameters...")
    spread_mean, spread_std = compute_is_distribution(model_id)

    # Step 2: 构建全量估值数据
    print("\n[Step 2] Building valuation dataframe...")
    df, model_params = build_valuation_dataframe(model_id, start_date, end_date)

    if df.empty:
        print("[ERROR] No valuation data produced!")
        return pd.DataFrame()

    print(f"  Data range: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} rows")

    # Step 3: 计算 Z-Score
    print("\n[Step 3] Computing spread Z-Score...")
    df['spread_zscore'] = compute_spread_zscore(df['valuation_spread'], spread_mean, spread_std)

    # Step 4: 计算目标敞口
    print("\n[Step 4] Computing target exposure...")
    df['target_exposure'] = compute_target_exposure(df['spread_zscore'])

    # Step 5: 统计摘要
    print("\n[Step 5] Summary statistics:")
    print(f"  Valuation Spread:  mean={df['valuation_spread'].mean():.6f}, "
          f"std={df['valuation_spread'].std():.6f}, "
          f"min={df['valuation_spread'].min():.6f}, max={df['valuation_spread'].max():.6f}")
    print(f"  Spread Z-Score:    mean={df['spread_zscore'].mean():.4f}, "
          f"std={df['spread_zscore'].std():.4f}, "
          f"min={df['spread_zscore'].min():.4f}, max={df['spread_zscore'].max():.4f}")
    print(f"  Target Exposure:   mean={df['target_exposure'].mean():.4f}, "
          f"std={df['target_exposure'].std():.4f}, "
          f"min={df['target_exposure'].min():.4f}, max={df['target_exposure'].max():.4f}")

    long_pct = (df['target_exposure'] > 0).sum() / len(df) * 100
    short_pct = (df['target_exposure'] < 0).sum() / len(df) * 100
    flat_pct = (df['target_exposure'] == 0).sum() / len(df) * 100
    print(f"  Exposure distribution: Long={long_pct:.1f}%, Flat={flat_pct:.1f}%, Short={short_pct:.1f}%")

    # Step 6: 最新状态
    latest = df.iloc[-1]
    print(f"\n  Latest valuation ({df.index[-1].date()}):")
    print(f"    Market Price  = {latest['market_price']:.2f}")
    print(f"    Fair Value    = {latest['fair_value']:.2f}")
    print(f"    Spread        = {latest['valuation_spread']:.6f} ({latest['valuation_spread']*100:.3f}%)")
    print(f"    Z-Score       = {latest['spread_zscore']:.4f}")
    print(f"    Target Exp    = {latest['target_exposure']:.4f}")

    # Step 7: 落库
    print("\n[Step 6] Saving to macro_valuation_daily...")
    save_valuation_daily(df, model_id, symbol)

    print("\n" + "=" * 78)
    print("Target Exposure Engine Complete!")
    print("=" * 78)

    return df


def run_engine_v6(symbol='GCUSD', start_date=None, end_date=None):
    """
    运行 V6 (Macro-Trend 共振版) Target Exposure 引擎

    Args:
        symbol: 标的符号
        start_date: 起始日期 (None=全量)
        end_date: 结束日期 (None=全量)

    Returns:
        DataFrame: 完整估值结果
    """
    print("=" * 78)
    print(f"Target Exposure Engine V6 (Macro-Trend Resonance) / {symbol}")
    print("=" * 78)

    model = GoldMacroTrendV6()
    df = model.calculate_target_exposure(start_date=start_date, end_date=end_date)

    if df.empty:
        print("[ERROR] No V6 data produced!")
        return pd.DataFrame()

    print("\n[DB] Saving V6 results to macro_valuation_daily...")
    save_valuation_daily(df, 'gold_macro_v6', symbol)

    print("\n" + "=" * 78)
    print("V6 Target Exposure Engine Complete!")
    print("=" * 78)

    return df


def run_engine_v7(symbol='GCUSD', start_date=None, end_date=None):
    """
    运行 V7 (Macro Veto Edition) Target Exposure 引擎

    Args:
        symbol: 标的符号
        start_date: 起始日期 (None=全量)
        end_date: 结束日期 (None=全量)

    Returns:
        DataFrame: 完整估值结果
    """
    print("=" * 78)
    print(f"Target Exposure Engine V7 (Macro Veto Edition) / {symbol}")
    print("=" * 78)

    model = GoldMacroTrendV7()
    df = model.calculate_target_exposure(start_date=start_date, end_date=end_date)

    if df.empty:
        print("[ERROR] No V7 data produced!")
        return pd.DataFrame()

    print("\n[DB] Saving V7 results to macro_valuation_daily...")
    save_valuation_daily(df, 'gold_macro_v7', symbol)

    print("\n" + "=" * 78)
    print("V7 Target Exposure Engine Complete!")
    print("=" * 78)

    return df


def run_engine_v8(symbol='GCUSD', start_date=None, end_date=None):
    """
    运行 V8 (Multi-Pillar Scoring Edition) Target Exposure 引擎

    Args:
        symbol: 标的符号
        start_date: 起始日期 (None=全量)
        end_date: 结束日期 (None=全量)

    Returns:
        DataFrame: 完整估值结果
    """
    print("=" * 78)
    print(f"Target Exposure Engine V8 (Multi-Pillar Scoring) / {symbol}")
    print("=" * 78)

    model = GoldMacroTrendV8()
    df = model.calculate_target_exposure(start_date=start_date, end_date=end_date)

    if df.empty:
        print("[ERROR] No V8 data produced!")
        return pd.DataFrame()

    print("\n[DB] Saving V8 results to macro_valuation_daily...")
    save_valuation_daily(df, 'gold_macro_v8', symbol)

    print("\n" + "=" * 78)
    print("V8 Target Exposure Engine Complete!")
    print("=" * 78)

    return df


if __name__ == '__main__':
    run_engine()
