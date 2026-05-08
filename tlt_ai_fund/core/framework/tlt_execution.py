#!/usr/bin/env python3
"""
Execution - TLT 纯防守执行器 (v10 - 大道至简)

核心哲学:
  彻底删除所有追涨代码 (bull_floor, bull_run)！
  宏观估值引擎已能自适应给仓位，执行层只做纯粹的防守。

唯一霸权: 向下熔断 (Bear Trap)
  价格跌破 SMA50 和 SMA200 → 强制清仓保本金
  不管宏观觉得多便宜，市场在疯狂抛售时必须切断手腕
"""

import numpy as np
import pandas as pd


class TltExecution:
    """
    TLT 纯防守执行器 (v10)

    输出列:
      sma50, sma200: 均线 (基于 adj_close)
      bear_trap: 向下熔断标志
      trend_regime: 趋势区间标签
      target_exposure: 最终目标敞口
    """

    def __init__(self, sma_standard: int = 50, sma_slow: int = 200):
        self.sma_standard = sma_standard
        self.sma_slow = sma_slow

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df['target_exposure'] = df['tlt_core_signal'].copy()

        price = df['adj_close']
        df['sma50'] = price.rolling(window=self.sma_standard, min_periods=self.sma_standard).mean()
        df['sma200'] = price.rolling(window=self.sma_slow, min_periods=self.sma_slow).mean()

        df['bear_trap'] = (price < df['sma50']) & (price < df['sma200'])

        # ── 唯一霸权: 向下熔断 ──
        df.loc[df['bear_trap'], 'target_exposure'] = 0.0

        df['target_exposure'] = np.clip(df['target_exposure'], 0.0, 1.0)
        df['target_exposure'] = df['target_exposure'].fillna(0.0)

        # ── 趋势区间标签 ──
        df['trend_regime'] = 'normal'
        df.loc[df['bear_trap'], 'trend_regime'] = 'bear_trap'

        return df

    def __repr__(self):
        return f"TltExecution(sma=[{self.sma_standard}, {self.sma_slow}], mode=Bear_Trap_Only_v10)"
