#!/usr/bin/env python3
"""
Core Anchor - 绝对底座因子 (SMA 投票版)

底座代表黄金永恒的物理重力:
  - 实际利率 (TIPS/DFII10): 持有黄金的机会成本
  - 美元指数 (DXY): 黄金的计价货币
  - 央行流动性 (WALCL/M2): 法币信用的反面

核心逻辑 (均线动量投票):
  1. 计算每个宏观变量的 60 日 SMA
  2. 独立投票: 当前值 vs SMA60, 输出 ±1
  3. 三票等权平均, 输出 core_signal ∈ {-1.0, -0.33, 0.33, 1.0}

投票规则:
  vote_tips  = -1 if DFII10 > SMA60(DFII10), else +1  (利率高于均线=利空)
  vote_dxy   = -1 if DXY > SMA60(DXY),       else +1  (美元高于均线=利空)
  vote_walcl = +1 if WALCL > SMA60(WALCL),    else -1  (扩表高于均线=利多)
  core_signal = (vote_tips + vote_dxy + vote_walcl) / 3.0
"""

import numpy as np
import pandas as pd


class CoreMacroAnchor:
    """
    绝对底座 - SMA 均线投票机制

    核心思想:
      Z-Score(Diff) 抹杀了稳定的长周期趋势, 因为差分运算消除了水平信息。
      SMA 投票直接保留"当前值 vs 均值"的关系, 能同时捕捉:
        - 趋势方向 (值在均线上方还是下方)
        - 趋势持续性 (均线本身在缓慢跟随)

    输出:
      core_signal ∈ {-1.0, -0.33, 0.33, 1.0} (四个离散状态)
    """

    def __init__(self, sma_window: int = 60):
        self.sma_window = sma_window
        self.name = 'core_anchor'

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算 SMA 投票底座信号

        Args:
            data: DataFrame, 必须包含 dfii10, dtwexbgs 列, 可选 walcl 列

        Returns:
            DataFrame: 新增列 vote_tips, vote_dxy, vote_walcl, core_signal
        """
        df = data.copy()

        sma_tips = df['dfii10'].rolling(
            window=self.sma_window, min_periods=self.sma_window
        ).mean()
        sma_dxy = df['dtwexbgs'].rolling(
            window=self.sma_window, min_periods=self.sma_window
        ).mean()

        df['vote_tips'] = np.where(
            df['dfii10'] > sma_tips, -1.0, 1.0
        )
        df['vote_dxy'] = np.where(
            df['dtwexbgs'] > sma_dxy, -1.0, 1.0
        )

        if 'walcl' in df.columns and df['walcl'].notna().any():
            sma_walcl = df['walcl'].rolling(
                window=self.sma_window, min_periods=self.sma_window
            ).mean()
            df['vote_walcl'] = np.where(
                df['walcl'] > sma_walcl, 1.0, -1.0
            )
        else:
            df['vote_walcl'] = 0.0

        df['core_signal'] = (
            df['vote_tips'] + df['vote_dxy'] + df['vote_walcl']
        ) / 3.0

        df.loc[df['dfii10'].isna() | df['dtwexbgs'].isna(), 'core_signal'] = np.nan

        return df

    def __repr__(self):
        return f"CoreMacroAnchor(sma_window={self.sma_window}, mode=SMA_Voting)"
