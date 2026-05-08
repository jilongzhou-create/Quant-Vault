#!/usr/bin/env python3
"""
Core Anchor - SPY 美股宏观底座因子 (SMA 投票版)

三大宏观支柱:
  1. 增长锚 (INDPRO): 工业生产指数与 SMA60 比较
     - INDPRO > SMA60 (增长扩张) → vote_growth = +1.0 (利好美股)
     - INDPRO < SMA60 (增长收缩) → vote_growth = -1.0 (利空美股)

  2. 就业锚 (ICSA): 初请失业金人数与 SMA60 比较 (注意：越低越好)
     - ICSA < SMA60 (就业强劲) → vote_employment = +1.0 (利好美股)
     - ICSA > SMA60 (就业疲软) → vote_employment = -1.0 (利空美股)

  3. 流动性锚 (Net_Liquidity = WALCL - WTREGEN - RRPONTSYD):
     - Net_Liquidity > SMA60 (流动性充裕) → vote_liquidity = +1.0 (利好美股)
     - Net_Liquidity < SMA60 (流动性收紧) → vote_liquidity = -1.0 (利空美股)

最终信号:
  spy_core_signal = (vote_growth + vote_employment + vote_liquidity) / 3.0
  ∈ {-1.0, -0.33, 0.33, 1.0}
"""

import numpy as np
import pandas as pd


class SpyCoreAnchor:
    """
    SPY 宏观底座 - SMA 均线投票机制

    输出:
      spy_core_signal ∈ {-1.0, -0.33, 0.33, 1.0} (四个离散状态)
    """

    def __init__(self, sma_window: int = 60):
        self.sma_window = sma_window
        self.name = 'spy_core_anchor'

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算 SMA 投票底座信号

        Args:
            data: DataFrame, 必须包含 indpro, icsa, walcl, wtregen, rrpontsyd 列
                  (或已合成 net_liquidity 列)

        Returns:
            DataFrame: 新增列 vote_growth, vote_employment, vote_liquidity, spy_core_signal
        """
        df = data.copy()

        if 'net_liquidity' not in df.columns:
            required = ['walcl', 'wtregen', 'rrpontsyd']
            if all(col in df.columns for col in required):
                df['net_liquidity'] = df['walcl'] - df['wtregen'] - df['rrpontsyd']
            else:
                missing = [c for c in required if c not in df.columns]
                raise ValueError(f"缺少流动性计算所需列: {missing}")

        sma_indpro = df['indpro'].rolling(
            window=self.sma_window, min_periods=self.sma_window
        ).mean()
        sma_icsa = df['icsa'].rolling(
            window=self.sma_window, min_periods=self.sma_window
        ).mean()
        sma_net_liq = df['net_liquidity'].rolling(
            window=self.sma_window, min_periods=self.sma_window
        ).mean()

        df['vote_growth'] = np.where(
            df['indpro'] > sma_indpro, 1.0, -1.0
        )

        df['vote_employment'] = np.where(
            df['icsa'] < sma_icsa, 1.0, -1.0
        )

        df['vote_liquidity'] = np.where(
            df['net_liquidity'] > sma_net_liq, 1.0, -1.0
        )

        df['spy_core_signal'] = (
            df['vote_growth'] + df['vote_employment'] + df['vote_liquidity']
        ) / 3.0

        df.loc[
            df['indpro'].isna() | df['icsa'].isna() | df['net_liquidity'].isna(),
            'spy_core_signal'
        ] = np.nan

        return df

    def __repr__(self):
        return f"SpyCoreAnchor(sma_window={self.sma_window}, mode=SMA_Voting)"
