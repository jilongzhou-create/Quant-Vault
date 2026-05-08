#!/usr/bin/env python3
"""
Execution - SPY 敞口映射执行器 (长牛+均值回归版)

敞口映射矩阵 (优先判定顺序):

  第一步 [超级抄底豁免]:
    RSI_14 < 30 → 无视均线和宏观信号, 强制 target_exposure = 0.75

  第二步 [常规状态判定]:
    Condition A (绝对多头): Price > SMA200 且 Price > SMA50
      - spy_core_signal > 0  → 敞口 = 1.0
      - spy_core_signal <= 0 → 敞口 = 0.5 (防踏空豁免)

    Condition B (多头震荡/回调): Price > SMA200 且 Price <= SMA50
      - spy_core_signal > 0  → 敞口 = 0.75
      - spy_core_signal <= 0 → 敞口 = 0.25

    Condition C (熊市或空头排列): Price <= SMA200
      - spy_core_signal > 0  → 敞口 = 0.25
      - spy_core_signal <= 0 → 敞口 = 0.0 (向下熔断, 空仓吃利息)

铁律:
  - 绝对纯多头: target_exposure ∈ [0.0, 1.0]
  - 禁止 Vol-Targeting、禁止回撤冷却降仓
  - 数据复权: 价格必须基于 adjClose
"""

import numpy as np
import pandas as pd


class SpyExecution:
    """
    SPY 敞口映射执行器 (长牛+均值回归版)

    输出列:
      sma_50, sma_200: 均线
      rsi_14: 14日 RSI
      trend_regime: 趋势区间标签
      target_exposure: 最终目标敞口 [0.0, 1.0]
    """

    def __init__(self, sma_standard: int = 50, sma_slow: int = 200,
                 rsi_window: int = 14, rsi_oversold: float = 30.0):
        self.sma_standard = sma_standard
        self.sma_slow = sma_slow
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold

    def calculate(self, data: pd.DataFrame,
                 core_signal: pd.Series) -> pd.DataFrame:
        df = data.copy()
        df['spy_core_signal'] = core_signal

        price = df['market_price']

        df['sma_50'] = price.rolling(
            window=self.sma_standard, min_periods=self.sma_standard
        ).mean()
        df['sma_200'] = price.rolling(
            window=self.sma_slow, min_periods=self.sma_slow
        ).mean()

        df['rsi_14'] = self._calc_rsi(price, self.rsi_window)

        signal = df['spy_core_signal']

        above_sma200 = price > df['sma_200']
        above_sma50 = price > df['sma_50']
        rsi_oversold = df['rsi_14'] < self.rsi_oversold

        cond_a = above_sma200 & above_sma50
        cond_b = above_sma200 & ~above_sma50
        cond_c = ~above_sma200

        sig_pos = signal > 0
        sig_neg = ~sig_pos

        df['trend_regime'] = 'neutral'

        df.loc[rsi_oversold, 'trend_regime'] = 'oversold_dip_buy'
        df.loc[cond_a & ~rsi_oversold & sig_pos, 'trend_regime'] = 'abs_uptrend_bull'
        df.loc[cond_a & ~rsi_oversold & sig_neg, 'trend_regime'] = 'abs_uptrend_hedge'
        df.loc[cond_b & ~rsi_oversold & sig_pos, 'trend_regime'] = 'pullback_bull'
        df.loc[cond_b & ~rsi_oversold & sig_neg, 'trend_regime'] = 'pullback_hedge'
        df.loc[cond_c & ~rsi_oversold & sig_pos, 'trend_regime'] = 'bear_rally'
        df.loc[cond_c & ~rsi_oversold & sig_neg, 'trend_regime'] = 'bear_circuit'

        exposure = pd.Series(np.nan, index=df.index, dtype=float)

        # 第一步: 超级抄底豁免
        exposure.loc[rsi_oversold] = 0.75

        # 第二步: Condition A (绝对多头)
        exposure.loc[cond_a & ~rsi_oversold & sig_pos] = 1.0
        exposure.loc[cond_a & ~rsi_oversold & sig_neg] = 0.5

        # Condition B (多头震荡/回调)
        exposure.loc[cond_b & ~rsi_oversold & sig_pos] = 0.75
        exposure.loc[cond_b & ~rsi_oversold & sig_neg] = 0.25

        # Condition C (熊市或空头排列)
        exposure.loc[cond_c & ~rsi_oversold & sig_pos] = 0.25
        exposure.loc[cond_c & ~rsi_oversold & sig_neg] = 0.0

        df['target_exposure'] = exposure.clip(0.0, 1.0)

        return df

    @staticmethod
    def _calc_rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def __repr__(self):
        return (f"SpyExecution(sma=[{self.sma_standard}, {self.sma_slow}], "
                f"rsi={self.rsi_window}, oversold={self.rsi_oversold}, "
                f"mode=LongBull_MeanReversion)")
