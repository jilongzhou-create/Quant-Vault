#!/usr/bin/env python3
"""
Execution - 自适应执行器 (离散得分版)

配合 CoreMacroAnchor 的 SMA 投票机制, core_signal 值域为:
  {-1.0, -0.33, 0.33, 1.0} 四个离散状态

执行规则:
  Score == 1.0  (极强顺风): Price > SMA50 且 SMA50 > SMA200 → 仓位 1.0
  Score == 0.33 (温和顺风): Price > SMA50 且 SMA50 > SMA200 → 仓位 0.75
  Score == -0.33(温和逆风): 硬否决, 强制仓位 0.0
  Score == -1.0 (极度恶劣): 硬否决, 强制仓位 0.0

趋势破位规则:
  任何情况下, 只要 Price < SMA200 或 SMA50 < SMA200, 强制仓位 0.0
"""

import numpy as np
import pandas as pd


class AdaptiveExecution:
    """
    自适应执行器 (离散得分版)

    输出列:
      sma_medium: 中速均线 (SMA50)
      sma_slow: 慢速均线 (SMA200)
      trend_intact: 趋势完好标志 (Price > SMA50 且 SMA50 > SMA200)
      trend_break: 趋势破位标志 (Price < SMA200 或 SMA50 < SMA200)
      score_regime: 得分区间标签
      target_exposure: 最终目标敞口
    """

    def __init__(self, sma_standard: int = 50, sma_slow: int = 200):
        self.sma_standard = sma_standard
        self.sma_slow = sma_slow

    def calculate(self, data: pd.DataFrame,
                  total_score: pd.Series) -> pd.DataFrame:
        """
        执行自适应敞口映射

        Args:
            data: DataFrame, 必须包含 market_price 列
            total_score: 合成总分 Series (来自 CoreMacroAnchor)

        Returns:
            DataFrame: 新增趋势和敞口相关列
        """
        df = data.copy()
        df['total_score'] = total_score

        price = df['market_price']

        df['sma_medium'] = price.rolling(
            window=self.sma_standard, min_periods=self.sma_standard
        ).mean()
        df['sma_slow'] = price.rolling(
            window=self.sma_slow, min_periods=self.sma_slow
        ).mean()

        df['trend_intact'] = (
            (price > df['sma_medium']) & (df['sma_medium'] > df['sma_slow'])
        ).astype(float)

        df['trend_break'] = (
            (price < df['sma_slow']) | (df['sma_medium'] < df['sma_slow'])
        ).astype(float)

        score = df['total_score']

        extreme_tail = (score > 0.5)
        mild_tail = (score > 0) & (score <= 0.5)
        mild_head = (score < 0) & (score >= -0.5)
        hard_veto = (score < -0.5)

        df['score_regime'] = 'neutral'
        df.loc[extreme_tail, 'score_regime'] = 'extreme_tailwind'
        df.loc[mild_tail, 'score_regime'] = 'mild_tailwind'
        df.loc[mild_head, 'score_regime'] = 'mild_headwind'
        df.loc[hard_veto, 'score_regime'] = 'hard_veto'

        exposure = pd.Series(0.0, index=df.index)

        exposure.loc[extreme_tail] = (
            df.loc[extreme_tail, 'trend_intact'] * 1.0
        )

        exposure.loc[mild_tail] = (
            df.loc[mild_tail, 'trend_intact'] * 0.75
        )

        exposure.loc[mild_head] = 0.0
        exposure.loc[hard_veto] = 0.0

        exposure.loc[df['trend_break'] == 1] = 0.0

        df['target_exposure'] = exposure.clip(0.0, 1.0)

        return df

    def __repr__(self):
        return (f"AdaptiveExecution(sma=[{self.sma_standard}, "
                f"{self.sma_slow}], mode=Discrete)")
