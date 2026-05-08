#!/usr/bin/env python3
"""
GoldValuationModel - 黄金宏观估值模型实现类 (Smart Trend Edition)

范式进化:
  V1 (纯均值回归): TargetExposure = -clip(Z / z_cap, -1, 1)
     问题: DW=0.024, 残差高度自相关, 在大牛市中持续做空挨打

  V2 (缩减敞口):   FinalExposure = RawExposure × ScalingScore
     问题: ScalingScore→0 时敞口归零, 错过牛市收益

  V2b (趋势融合):  FinalExposure = RawExposure × Scale + Trend × (1-Scale)
     问题: 趋势信号频繁翻转导致敞口震荡, 换手率暴增

  V3 (趋势主导):   趋势强时跟趋势, 趋势弱时用估值
     FinalExposure = (1-trend_weight) × RawExposure + trend_weight × TrendSignal
     trend_weight = clip(|trend_signal| / trend_threshold, 0, 1)
     核心逻辑: 趋势明确时顺势, 趋势模糊时回归
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
from isolated_macro.core.valuation_model import (
    load_model,
    load_market_data,
    load_fred_raw_series,
    load_fred_factor_series,
    load_macro_factor,
    compute_fair_value,
)

ROLLING_WINDOW = 1260
Z_SCORE_CAP = 2.0
SMA_PERIOD = 200
TREND_CAP = 0.15
TREND_THRESHOLD = 0.3


class GoldValuationModel:
    """
    黄金宏观估值模型 (Smart Trend Edition)

    用法:
        model = GoldValuationModel(model_id='gold_macro_v1')
        df_result = model.calculate_target_exposure(start_date='2007-01-01', end_date='2019-12-31')
    """

    def __init__(self, model_id='gold_macro_v1', rolling_window=ROLLING_WINDOW,
                 z_cap=Z_SCORE_CAP, sma_period=SMA_PERIOD, trend_cap=TREND_CAP,
                 trend_threshold=TREND_THRESHOLD):
        self.model_id = model_id
        self.rolling_window = rolling_window
        self.z_cap = z_cap
        self.sma_period = sma_period
        self.trend_cap = trend_cap
        self.trend_threshold = trend_threshold
        self.model_params = load_model(model_id)
        self.data_mode = self.model_params.get('data_mode', 'raw_level')

        print(f"[GoldValuationModel] model_id={model_id}, data_mode={self.data_mode}")
        print(f"  intercept={self.model_params['intercept']:.6f}")
        print(f"  b_DFII10={self.model_params['b_DFII10']:.6f}")
        print(f"  b_lnDXY={self.model_params['b_lnDXY']:.6f}")
        print(f"  rolling_window={rolling_window}, z_cap={z_cap}")
        print(f"  sma_period={sma_period}, trend_cap={trend_cap}, trend_threshold={trend_threshold}")

    def _load_and_merge_data(self, start_date=None, end_date=None):
        """加载并合并行情 + 宏观数据"""
        df_gold = load_market_data('GCUSD', start_date, end_date)

        if self.data_mode == 'raw_level':
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

        return df

    def _compute_raw_exposure(self, df):
        """计算估值偏差 Z-Score 和 Raw Exposure (纯均值回归信号)"""
        df = compute_fair_value(self.model_params, df)

        df['spread_zscore'] = (
            df['valuation_spread'] - df['valuation_spread'].rolling(
                window=self.rolling_window, min_periods=60
            ).mean()
        ) / df['valuation_spread'].rolling(
            window=self.rolling_window, min_periods=60
        ).std()

        df['spread_zscore'] = df['spread_zscore'].fillna(0)

        df['raw_exposure'] = -np.clip(df['spread_zscore'] / self.z_cap, -1.0, 1.0)

        return df

    def _compute_trend_signal(self, df):
        """计算趋势代理信号 (200d SMA)"""
        df['sma_200'] = df['market_price'].rolling(window=self.sma_period, min_periods=60).mean()
        df['trend_dev'] = (df['market_price'] - df['sma_200']) / df['sma_200']
        df['trend_signal'] = np.clip(df['trend_dev'] / self.trend_cap, -1.0, 1.0)
        df['trend_signal'] = df['trend_signal'].fillna(0)
        return df

    def _compute_all_exposures(self, df):
        """
        计算所有版本的敞口, 便于回测对比

        V1: 纯均值回归
            target_exposure_v1 = raw_exposure

        V2: 缩减敞口 (Scale-Down)
            scaling_score = clip(1 + sign(raw) * trend, 0, 1)
            target_exposure_v2 = raw * scaling_score

        V2b: 趋势融合 (Fusion)
            target_exposure_v2b = raw * scale + trend * (1 - scale)

        V3: 趋势主导 (Trend-Dominant)
            trend_weight = clip(|trend_signal| / threshold, 0, 1)
            target_exposure_v3 = (1 - trend_weight) * raw + trend_weight * trend
            核心逻辑: 趋势明确时顺势, 趋势模糊时回归
        """
        raw = df['raw_exposure']
        trend = df['trend_signal']

        # V1: 纯均值回归
        df['target_exposure_v1'] = raw

        # V2: 缩减敞口
        raw_sign = np.sign(raw)
        raw_sign = raw_sign.replace(0, 1)
        scaling_score = np.clip(1.0 + raw_sign * trend, 0.0, 1.0)
        df['scaling_score'] = scaling_score
        df['target_exposure_v2'] = raw * scaling_score

        # V2b: 趋势融合
        df['target_exposure_v2b'] = np.clip(
            raw * scaling_score + trend * (1.0 - scaling_score),
            -1.0, 1.0
        )

        # V3: 趋势主导
        trend_weight = np.clip(np.abs(trend) / self.trend_threshold, 0.0, 1.0)
        df['trend_weight'] = trend_weight
        df['target_exposure_v3'] = np.clip(
            (1.0 - trend_weight) * raw + trend_weight * trend,
            -1.0, 1.0
        )

        # V4: Long-Only V3 (黄金有结构性上行偏置, 永远不做空)
        df['target_exposure_v4'] = np.clip(df['target_exposure_v3'], 0.0, 1.0)

        # V5: Pure Trend (完全不用估值, 纯趋势跟踪)
        df['target_exposure_v5'] = trend

        # 默认使用 V3
        df['target_exposure'] = df['target_exposure_v3']

        return df

    def calculate_target_exposure(self, start_date=None, end_date=None):
        """
        完整估值计算管线 (Smart Trend Edition)

        Returns:
            DataFrame: 包含所有估值、趋势、敞口列
        """
        df = self._load_and_merge_data(start_date, end_date)

        if df.empty:
            print("[ERROR] No data after merge!")
            return pd.DataFrame()

        df = self._compute_raw_exposure(df)
        df = self._compute_trend_signal(df)
        df = self._compute_all_exposures(df)

        print(f"\n[Valuation Result] {len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"  Valuation Spread: mean={df['valuation_spread'].mean():.6f}, "
              f"std={df['valuation_spread'].std():.6f}")
        print(f"  Spread Z-Score:   mean={df['spread_zscore'].mean():.4f}, "
              f"std={df['spread_zscore'].std():.4f}, "
              f"min={df['spread_zscore'].min():.4f}, max={df['spread_zscore'].max():.4f}")
        print(f"  Raw Exposure:     mean={df['raw_exposure'].mean():.4f}, "
              f"std={df['raw_exposure'].std():.4f}")
        print(f"  Trend Signal:     mean={df['trend_signal'].mean():.4f}, "
              f"std={df['trend_signal'].std():.4f}")
        print(f"  Trend Weight:     mean={df['trend_weight'].mean():.4f}, "
              f"std={df['trend_weight'].std():.4f}")

        for ver in ['v1', 'v2', 'v2b', 'v3', 'v4', 'v5']:
            col = f'target_exposure_{ver}'
            avg = df[col].mean()
            std = df[col].std()
            long_pct = (df[col] > 0).sum() / len(df) * 100
            short_pct = (df[col] < 0).sum() / len(df) * 100
            print(f"  {ver.upper():4s} Exposure:     mean={avg:+.4f}, std={std:.4f}, "
                  f"Long={long_pct:.0f}%, Short={short_pct:.0f}%")

        return df

    def print_zscore_exposure_series(self, df, n_tail=20):
        """打印 Z-Score 和 Exposure 时间序列尾部"""
        print(f"\n[Z-Score & Exposure] Last {n_tail} rows:")
        print(f"  {'Date':>12s}  {'Price':>10s}  {'ZScore':>7s}  "
              f"{'RawExp':>7s}  {'Trend':>6s}  {'TWeight':>7s}  "
              f"{'V1':>7s}  {'V2':>7s}  {'V2b':>7s}  {'V3':>7s}")
        print("-" * 105)
        for ts, row in df.tail(n_tail).iterrows():
            print(f"  {ts.date()}  {row['market_price']:>10.2f}  {row['spread_zscore']:>7.2f}  "
                  f"{row['raw_exposure']:>7.4f}  {row['trend_signal']:>6.2f}  "
                  f"{row['trend_weight']:>7.4f}  "
                  f"{row['target_exposure_v1']:>7.4f}  {row['target_exposure_v2']:>7.4f}  "
                  f"{row['target_exposure_v2b']:>7.4f}  {row['target_exposure_v3']:>7.4f}")


class GoldMacroTrendV6:
    """
    V6: Macro-Trend 共振版

    彻底放弃 OLS 预测绝对价格, 改用"宏观变量变化率 + 双均线趋势"直接生成目标敞口。
    严格遵守 Long-Only (0 到 1 之间)。

    趋势计算 (S_trend):
      SMA50 = rolling(50).mean()
      SMA200 = rolling(200).mean()
      trend_flag = 1  当 Price > SMA200 且 SMA50 > SMA200
      trend_flag = 0  其他

    宏观计算 (S_macro):
      dfii10_diff = DFII10 - DFII10.shift(60)   (实际利率 60 日变化)
      dxy_diff = ln(DTWEXBGS) - ln(DTWEXBGS).shift(60)  (美元指数 60 日对数变化)
      macro_flag = 1   当 dfii10_diff < 0 且 dxy_diff < 0  (利率降+美元弱 = 强利好黄金)
      macro_flag = -1  当 dfii10_diff > 0 且 dxy_diff > 0  (利率升+美元强 = 强利空黄金)
      macro_flag = 0   其他 (宏观信号不一致)

    敞口合成 (Long-Only):
      trend_flag == 0: target_exposure = 0.0  (趋势不成立, 空仓)
      trend_flag == 1:
        macro_flag == 1:  target_exposure = 1.0   (趋势+宏观共振, 满仓)
        macro_flag == 0:  target_exposure = 0.75  (趋势成立但宏观中性, 3/4 仓)
        macro_flag == -1: target_exposure = 0.5   (趋势成立但宏观逆风, 半仓)

    用法:
        model = GoldMacroTrendV6()
        df_result = model.calculate_target_exposure(start_date='2007-01-01', end_date='2019-12-31')
    """

    def __init__(self, sma_fast=50, sma_slow=200, macro_lookback=60):
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.macro_lookback = macro_lookback
        self.data_mode = 'raw_level'

        print(f"[GoldMacroTrendV6] sma_fast={sma_fast}, sma_slow={sma_slow}, macro_lookback={macro_lookback}")
        print(f"  Long-Only: target_exposure in [0, 1]")
        print(f"  No OLS beta coefficients needed")

    def _load_and_merge_data(self, start_date=None, end_date=None):
        """加载并合并行情 + 宏观数据"""
        df_gold = load_market_data('GCUSD', start_date, end_date)
        df_dfii10 = load_fred_raw_series('DFII10', start_date, end_date)
        df_dtwexbgs = load_fred_raw_series('DTWEXBGS', start_date, end_date)

        df = df_gold.copy()
        if not df_dfii10.empty:
            df = df.join(df_dfii10[['dfii10']], how='left')
        if not df_dtwexbgs.empty:
            df = df.join(df_dtwexbgs[['dtwexbgs']], how='left')

        df = df.sort_index()
        df = df.ffill()
        df = df.dropna(subset=['market_price', 'dfii10', 'dtwexbgs'])

        return df

    def _compute_trend_flag(self, df):
        """计算双均线趋势信号"""
        df['sma_50'] = df['market_price'].rolling(
            window=self.sma_fast, min_periods=self.sma_fast
        ).mean()
        df['sma_200'] = df['market_price'].rolling(
            window=self.sma_slow, min_periods=self.sma_slow
        ).mean()

        above_sma200 = df['market_price'] > df['sma_200']
        golden_cross = df['sma_50'] > df['sma_200']

        df['trend_flag'] = 0
        df.loc[above_sma200 & golden_cross, 'trend_flag'] = 1

        df['trend_flag'] = df['trend_flag'].astype(float)
        return df

    def _compute_macro_flag(self, df):
        """计算宏观变化率信号"""
        df['dfii10_diff'] = df['dfii10'] - df['dfii10'].shift(self.macro_lookback)
        df['ln_dxy'] = np.log(df['dtwexbgs'])
        df['dxy_diff'] = df['ln_dxy'] - df['ln_dxy'].shift(self.macro_lookback)

        df['macro_flag'] = 0.0

        both_negative = (df['dfii10_diff'] < 0) & (df['dxy_diff'] < 0)
        both_positive = (df['dfii10_diff'] > 0) & (df['dxy_diff'] > 0)

        df.loc[both_negative, 'macro_flag'] = 1.0
        df.loc[both_positive, 'macro_flag'] = -1.0

        return df

    def _compute_target_exposure(self, df):
        """合成 V6 目标敞口 (Long-Only)"""
        df['target_exposure_v6'] = 0.0

        trend_on = df['trend_flag'] == 1

        macro_bull = trend_on & (df['macro_flag'] == 1)
        macro_neutral = trend_on & (df['macro_flag'] == 0)
        macro_bear = trend_on & (df['macro_flag'] == -1)

        df.loc[macro_bull, 'target_exposure_v6'] = 1.0
        df.loc[macro_neutral, 'target_exposure_v6'] = 0.75
        df.loc[macro_bear, 'target_exposure_v6'] = 0.5

        return df

    def calculate_target_exposure(self, start_date=None, end_date=None):
        """
        完整 V6 计算管线

        Returns:
            DataFrame: 包含 market_price, trend_flag, macro_flag, target_exposure_v6
        """
        df = self._load_and_merge_data(start_date, end_date)

        if df.empty:
            print("[ERROR] No data after merge!")
            return pd.DataFrame()

        df = self._compute_trend_flag(df)
        df = self._compute_macro_flag(df)
        df = self._compute_target_exposure(df)

        df['fair_value'] = np.nan
        df['valuation_spread'] = np.nan
        df['spread_zscore'] = np.nan
        df['target_exposure'] = df['target_exposure_v6']

        valid = df['target_exposure_v6'].notna()
        n_total = len(df)
        n_valid = valid.sum()

        print(f"\n[V6 Result] {n_total} rows, {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"  Trend Flag:       Bull={int((df['trend_flag']==1).sum())}d, "
              f"Off={int((df['trend_flag']==0).sum())}d")
        print(f"  Macro Flag:       Bull={int((df['macro_flag']==1).sum())}d, "
              f"Neutral={int((df['macro_flag']==0).sum())}d, "
              f"Bear={int((df['macro_flag']==-1).sum())}d")

        exp = df.loc[valid, 'target_exposure_v6']
        print(f"  Target Exposure:  mean={exp.mean():.4f}, std={exp.std():.4f}, "
              f"min={exp.min():.4f}, max={exp.max():.4f}")

        full_pct = (exp == 1.0).sum() / n_valid * 100
        three_q_pct = ((exp > 0.5) & (exp < 1.0)).sum() / n_valid * 100
        half_pct = (exp == 0.5).sum() / n_valid * 100
        zero_pct = (exp == 0.0).sum() / n_valid * 100
        print(f"  Exposure dist:    Full={full_pct:.1f}%, 3/4={three_q_pct:.1f}%, "
              f"Half={half_pct:.1f}%, Zero={zero_pct:.1f}%")

        return df


class GoldMacroTrendV7:
    """
    V7: Macro-Trend 共振版 (Macro Veto Edition)

    在 V6 基础上的关键改进:
      1. 增强趋势过滤: Price > SMA50 AND SMA50 > SMA200
         (V6 是 Price > SMA200 AND SMA50 > SMA200, V7 要求价格必须在快线上方)
      2. 宏观一票否决 (Macro Veto): macro_flag == -1 时强制空仓
         (V6 在宏观逆风时仍给 0.5 仓位, V7 认为宏观强逆风下的趋势是诱多假突破)
      3. 宏观中性时更保守: 0.5 仓位 (V6 给 0.75)

    敞口合成规则:
      macro_flag == -1:         target_exposure = 0.0  (宏观否决, 强制空仓)
      trend_flag == 0:          target_exposure = 0.0  (趋势破位, 空仓)
      trend_flag == 1 且 macro_flag != -1:
        macro_flag == 1:        target_exposure = 1.0  (趋势+宏观共振, 满仓)
        macro_flag == 0:        target_exposure = 0.5  (趋势成立但宏观中性, 半仓)

    用法:
        model = GoldMacroTrendV7()
        df_result = model.calculate_target_exposure(start_date='2007-01-01', end_date='2019-12-31')
    """

    def __init__(self, sma_fast=50, sma_slow=200, macro_lookback=60):
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.macro_lookback = macro_lookback
        self.data_mode = 'raw_level'

        print(f"[GoldMacroTrendV7] sma_fast={sma_fast}, sma_slow={sma_slow}, macro_lookback={macro_lookback}")
        print(f"  Trend: Price > SMA{self.sma_fast} AND SMA{self.sma_fast} > SMA{self.sma_slow}")
        print(f"  Macro Veto: macro_flag == -1 => target_exposure = 0.0")
        print(f"  Long-Only: target_exposure in [0, 1]")

    def _load_and_merge_data(self, start_date=None, end_date=None):
        df_gold = load_market_data('GCUSD', start_date, end_date)
        df_dfii10 = load_fred_raw_series('DFII10', start_date, end_date)
        df_dtwexbgs = load_fred_raw_series('DTWEXBGS', start_date, end_date)

        df = df_gold.copy()
        if not df_dfii10.empty:
            df = df.join(df_dfii10[['dfii10']], how='left')
        if not df_dtwexbgs.empty:
            df = df.join(df_dtwexbgs[['dtwexbgs']], how='left')

        df = df.sort_index()
        df = df.ffill()
        df = df.dropna(subset=['market_price', 'dfii10', 'dtwexbgs'])
        return df

    def _compute_trend_flag(self, df):
        df['sma_50'] = df['market_price'].rolling(
            window=self.sma_fast, min_periods=self.sma_fast
        ).mean()
        df['sma_200'] = df['market_price'].rolling(
            window=self.sma_slow, min_periods=self.sma_slow
        ).mean()

        above_sma50 = df['market_price'] > df['sma_50']
        golden_cross = df['sma_50'] > df['sma_200']

        df['trend_flag'] = 0
        df.loc[above_sma50 & golden_cross, 'trend_flag'] = 1
        df['trend_flag'] = df['trend_flag'].astype(float)
        return df

    def _compute_macro_flag(self, df):
        df['dfii10_diff'] = df['dfii10'] - df['dfii10'].shift(self.macro_lookback)
        df['ln_dxy'] = np.log(df['dtwexbgs'])
        df['dxy_diff'] = df['ln_dxy'] - df['ln_dxy'].shift(self.macro_lookback)

        df['macro_flag'] = 0.0
        both_negative = (df['dfii10_diff'] < 0) & (df['dxy_diff'] < 0)
        both_positive = (df['dfii10_diff'] > 0) & (df['dxy_diff'] > 0)
        df.loc[both_negative, 'macro_flag'] = 1.0
        df.loc[both_positive, 'macro_flag'] = -1.0
        return df

    def _compute_target_exposure(self, df):
        df['target_exposure_v7'] = 0.0

        trend_on = df['trend_flag'] == 1
        macro_veto = df['macro_flag'] == -1

        macro_bull = trend_on & ~macro_veto & (df['macro_flag'] == 1)
        macro_neutral = trend_on & ~macro_veto & (df['macro_flag'] == 0)

        df.loc[macro_bull, 'target_exposure_v7'] = 1.0
        df.loc[macro_neutral, 'target_exposure_v7'] = 0.5

        return df

    def calculate_target_exposure(self, start_date=None, end_date=None):
        df = self._load_and_merge_data(start_date, end_date)

        if df.empty:
            print("[ERROR] No data after merge!")
            return pd.DataFrame()

        df = self._compute_trend_flag(df)
        df = self._compute_macro_flag(df)
        df = self._compute_target_exposure(df)

        df['fair_value'] = np.nan
        df['valuation_spread'] = np.nan
        df['spread_zscore'] = np.nan
        df['target_exposure'] = df['target_exposure_v7']

        n_total = len(df)
        n_valid = n_total

        print(f"\n[V7 Result] {n_total} rows, {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"  Trend Flag:       Bull={int((df['trend_flag']==1).sum())}d, "
              f"Off={int((df['trend_flag']==0).sum())}d")
        print(f"  Macro Flag:       Bull={int((df['macro_flag']==1).sum())}d, "
              f"Neutral={int((df['macro_flag']==0).sum())}d, "
              f"Bear={int((df['macro_flag']==-1).sum())}d")

        veto_days = int(((df['macro_flag'] == -1) & (df['trend_flag'] == 1)).sum())
        print(f"  Macro Veto:       {veto_days}d (trend ON but vetoed by macro)")

        exp = df['target_exposure_v7']
        print(f"  Target Exposure:  mean={exp.mean():.4f}, std={exp.std():.4f}, "
              f"min={exp.min():.4f}, max={exp.max():.4f}")

        full_pct = (exp == 1.0).sum() / n_valid * 100
        half_pct = (exp == 0.5).sum() / n_valid * 100
        zero_pct = (exp == 0.0).sum() / n_valid * 100
        print(f"  Exposure dist:    Full={full_pct:.1f}%, Half={half_pct:.1f}%, Zero={zero_pct:.1f}%")

        return df


class GoldMacroTrendV8:
    """
    V8: Multi-Pillar Scoring Edition (多维状态打分版)

    V7 的问题: 单一维度"一票否决"(实际利率+美元)在 2022 防守完美,
    但 2023-2024 黄金脱锚时错杀——因为黄金脱离了美国宏观框架。

    V8 核心改造: 废除单一因子的绝对否决权, 引入"三大宏观支柱"积分制。
    只有当总分 < 0 时, 才触发一票否决。

    三大支柱:
      S_fin  (金融分):    DFII10_diff + DXY_diff → 利率/美元双降=+1, 双升=-1
      S_liq  (流动性分):  WALCL_roc + HY_zscore → 扩表/恐慌=+1, 缩表+无恐慌=-1
      S_mic  (微观/脱锚): SGE_pct → 东方极端抢筹=+1, 其他=0

    敞口合成:
      net_score = S_fin + S_liq + S_mic  (fillna(0) 后求和)
      trend_flag = 1 当 Price > SMA50 且 SMA50 > SMA200
      trend_flag == 0:         exposure = 0.0
      trend_flag == 1:
        net_score < 0:         exposure = 0.0  (多维综合否决)
        net_score == 0:        exposure = 0.5  (宏观拉锯)
        net_score > 0:         exposure = 1.0  (宏观顺风/脱锚共振)

    Lookahead Bias 防护:
      - 所有原始数据已通过 Data Adapter 加 lag 落库
      - 差分/ROC/Z-Score/分位数均为向后看 (backward-looking)
      - shift(1) 在回测引擎层保证昨天的信号今天才执行
    """

    def __init__(self, sma_fast=50, sma_slow=200, macro_lookback=60,
                 walcl_roc_threshold=0.05, walcl_shrink_threshold=-0.02,
                 hy_panic_threshold=2.0, sge_extreme_pct=0.90,
                 sge_lookback=252):
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.macro_lookback = macro_lookback
        self.walcl_roc_threshold = walcl_roc_threshold
        self.walcl_shrink_threshold = walcl_shrink_threshold
        self.hy_panic_threshold = hy_panic_threshold
        self.sge_extreme_pct = sge_extreme_pct
        self.sge_lookback = sge_lookback
        self.data_mode = 'raw_level'

        print(f"[GoldMacroTrendV8] Multi-Pillar Scoring Edition")
        print(f"  Trend: Price > SMA{sma_fast} AND SMA{sma_fast} > SMA{sma_slow}")
        print(f"  S_fin:  DFII10_diff(60d) + DXY_diff(60d)")
        print(f"  S_liq:  WALCL_roc(60d, >{walcl_roc_threshold} or <{walcl_shrink_threshold}) + HY_zscore(60d, >{hy_panic_threshold})")
        print(f"  S_mic:  SGE_pct({sge_lookback}d, >{sge_extreme_pct})")
        print(f"  Veto: net_score < 0 => exposure = 0.0")

    def _load_and_merge_data(self, start_date=None, end_date=None):
        df_gold = load_market_data('GCUSD', start_date, end_date)
        df_dfii10 = load_fred_raw_series('DFII10', start_date, end_date)
        df_dtwexbgs = load_fred_raw_series('DTWEXBGS', start_date, end_date)
        df_walcl = load_macro_factor('walcl', start_date, end_date)
        df_hy = load_macro_factor('bamlh0a0hym2', start_date, end_date)
        df_sge = load_macro_factor('sge_premium', start_date, end_date)

        df = df_gold.copy()
        if not df_dfii10.empty:
            df = df.join(df_dfii10[['dfii10']], how='left')
        if not df_dtwexbgs.empty:
            df = df.join(df_dtwexbgs[['dtwexbgs']], how='left')
        if not df_walcl.empty:
            df = df.join(df_walcl[['walcl']], how='left')
        if not df_hy.empty:
            df = df.join(df_hy[['bamlh0a0hym2']], how='left')
        if not df_sge.empty:
            df = df.join(df_sge[['sge_premium']], how='left')

        df = df.sort_index()
        df = df.ffill()
        df = df.dropna(subset=['market_price', 'dfii10', 'dtwexbgs'])

        return df

    def _compute_features(self, df):
        lb = self.macro_lookback

        df['dfii10_diff'] = df['dfii10'] - df['dfii10'].shift(lb)
        df['ln_dxy'] = np.log(df['dtwexbgs'])
        df['dxy_diff'] = df['ln_dxy'] - df['ln_dxy'].shift(lb)

        if 'walcl' in df.columns:
            df['walcl_roc'] = df['walcl'] / df['walcl'].shift(lb) - 1.0
        else:
            df['walcl_roc'] = np.nan

        if 'bamlh0a0hym2' in df.columns:
            hy_mean = df['bamlh0a0hym2'].rolling(window=lb, min_periods=30).mean()
            hy_std = df['bamlh0a0hym2'].rolling(window=lb, min_periods=30).std()
            df['hy_zscore'] = (df['bamlh0a0hym2'] - hy_mean) / (hy_std + 1e-9)
        else:
            df['hy_zscore'] = np.nan

        if 'sge_premium' in df.columns:
            df['sge_pct'] = df['sge_premium'].rolling(
                window=self.sge_lookback, min_periods=60
            ).rank(pct=True)
        else:
            df['sge_pct'] = np.nan

        return df

    def _compute_pillar_scores(self, df):
        df['S_fin'] = 0.0
        fin_bull = (df['dfii10_diff'] < 0) & (df['dxy_diff'] < 0)
        fin_bear = (df['dfii10_diff'] > 0) & (df['dxy_diff'] > 0)
        df.loc[fin_bull, 'S_fin'] = 1.0
        df.loc[fin_bear, 'S_fin'] = -1.0

        df['S_liq'] = 0.0
        liq_bull = (df['walcl_roc'] > self.walcl_roc_threshold) | (df['hy_zscore'] > self.hy_panic_threshold)
        liq_bear = (df['walcl_roc'] < self.walcl_shrink_threshold) & (df['hy_zscore'] < 0)
        df.loc[liq_bull, 'S_liq'] = 1.0
        df.loc[liq_bear, 'S_liq'] = -1.0

        df['S_mic'] = 0.0
        mic_bull = df['sge_pct'] > self.sge_extreme_pct
        df.loc[mic_bull, 'S_mic'] = 1.0

        df['S_fin'] = df['S_fin'].fillna(0)
        df['S_liq'] = df['S_liq'].fillna(0)
        df['S_mic'] = df['S_mic'].fillna(0)

        return df

    def _compute_trend_flag(self, df):
        df['sma_50'] = df['market_price'].rolling(
            window=self.sma_fast, min_periods=self.sma_fast
        ).mean()
        df['sma_200'] = df['market_price'].rolling(
            window=self.sma_slow, min_periods=self.sma_slow
        ).mean()

        above_sma50 = df['market_price'] > df['sma_50']
        golden_cross = df['sma_50'] > df['sma_200']

        df['trend_flag'] = 0
        df.loc[above_sma50 & golden_cross, 'trend_flag'] = 1
        df['trend_flag'] = df['trend_flag'].astype(float)
        return df

    def _compute_target_exposure(self, df):
        df['net_macro_score'] = df['S_fin'] + df['S_liq'] + df['S_mic']

        df['target_exposure_v8'] = 0.0

        trend_on = df['trend_flag'] == 1
        score_neg = df['net_macro_score'] < 0
        score_zero = df['net_macro_score'] == 0
        score_pos = df['net_macro_score'] > 0

        df.loc[trend_on & score_neg, 'target_exposure_v8'] = 0.0
        df.loc[trend_on & score_zero, 'target_exposure_v8'] = 0.5
        df.loc[trend_on & score_pos, 'target_exposure_v8'] = 1.0

        return df

    def calculate_target_exposure(self, start_date=None, end_date=None):
        df = self._load_and_merge_data(start_date, end_date)

        if df.empty:
            print("[ERROR] No data after merge!")
            return pd.DataFrame()

        df = self._compute_features(df)
        df = self._compute_pillar_scores(df)
        df = self._compute_trend_flag(df)
        df = self._compute_target_exposure(df)

        df['fair_value'] = np.nan
        df['valuation_spread'] = np.nan
        df['spread_zscore'] = np.nan
        df['target_exposure'] = df['target_exposure_v8']

        n_total = len(df)

        print(f"\n[V8 Result] {n_total} rows, {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"  Trend Flag:       Bull={int((df['trend_flag']==1).sum())}d, "
              f"Off={int((df['trend_flag']==0).sum())}d")

        for name, col in [('S_fin', 'S_fin'), ('S_liq', 'S_liq'), ('S_mic', 'S_mic')]:
            vc = df[col].value_counts().sort_index()
            parts = [f"{int(k)}:{int(v)}d" for k, v in vc.items()]
            print(f"  {name}:             {', '.join(parts)}")

        score_vc = df['net_macro_score'].value_counts().sort_index()
        parts = [f"{int(k)}:{int(v)}d" for k, v in score_vc.items()]
        print(f"  Net Score:        {', '.join(parts)}")

        exp = df['target_exposure_v8']
        full_pct = (exp == 1.0).sum() / n_total * 100
        half_pct = (exp == 0.5).sum() / n_total * 100
        zero_pct = (exp == 0.0).sum() / n_total * 100
        print(f"  Exposure dist:    Full={full_pct:.1f}%, Half={half_pct:.1f}%, Zero={zero_pct:.1f}%")

        veto_days = int(((df['net_macro_score'] < 0) & (df['trend_flag'] == 1)).sum())
        print(f"  Multi-Pillar Veto: {veto_days}d (trend ON but score < 0)")

        return df
