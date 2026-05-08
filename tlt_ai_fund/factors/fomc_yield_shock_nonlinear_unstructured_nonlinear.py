import numpy as np
import pandas as pd

class FomcYieldShockNonlinearFactor:
    """Policy Pivot Shock & Curve Dynamics (unstructured/nonlinear)

    逻辑: 捕捉美联储政策预期突变与收益率曲线极值的非线性交叉。当鸽派情绪突变或短端利率(dgs2)急剧下行且长短利差(t10y2y)急剧变陡(Bull Steepening)时，输出看多脉冲；当短端利率极端飙升后出现动量衰竭(Hawkish Exhaustion)时，遵循"极值+衰竭"的抄底铁律输出看多脉冲。反之输出看空脉冲。常态下严格保持零值休眠。
    数据: fomc_sentiment, dgs2, t10y2y
    触发: (fomc_diff5 Z-Score > 2.5) OR (dgs2_diff Z-Score < -2.0 AND 曲线急陡) OR (dgs2_diff Z-Score > 2.0 AND 跌破3日均线衰竭)
    输出: +1.0(确认降息/恐慌衰竭的看多脉冲), -1.0(确认加息的看空脉冲), 其余为 0.0
    """

    def __init__(self, momentum_window=5, zscore_window=63):
        self.name = 'fomc_yield_shock_nonlinear'
        self.momentum_window = momentum_window
        self.zscore_window = zscore_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index, name=self.name)
        
        req_cols = ['fomc_sentiment', 'dgs2', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return signal

        df = data[req_cols].ffill()

        # 1. FOMC Sentiment Marginal Change (Rule 3: 必须边际变化)
        # 1.0=极度鸽派, diff > 0 意味着边际转鸽
        fomc_diff = df['fomc_sentiment'].diff(self.momentum_window)
        fomc_std = fomc_diff.rolling(252).std().replace(0, np.nan)
        fomc_z = (fomc_diff - fomc_diff.rolling(252).mean()) / fomc_std
        fomc_z = fomc_z.fillna(0.0)

        # 2. DGS2 (Short End) Momentum & Exhaustion (Rule 2 & 3)
        dgs2_diff = df['dgs2'].diff(self.momentum_window)
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(self.zscore_window).mean()) / dgs2_diff.rolling(self.zscore_window).std()
        dgs2_z = dgs2_z.fillna(0.0)
        
        # Exhaustion / Continuation Indicators (二阶导衰竭确认)
        dgs2_falling = df['dgs2'] < df['dgs2'].rolling(3).mean()
        dgs2_rising = df['dgs2'] > df['dgs2'].rolling(3).mean()

        # 3. T10Y2Y (Yield Curve) Momentum (Rule 3)
        t10y2y_diff = df['t10y2y'].diff(self.momentum_window)
        t10y2y_z = (t10y2y_diff - t10y2y_diff.rolling(self.zscore_window).mean()) / t10y2y_diff.rolling(self.zscore_window).std()
        t10y2y_z = t10y2y_z.fillna(0.0)

        # --- Nonlinear Logic Cross ---
        
        # Pulse 1: FOMC Explicit Sentiment Shock (情绪得分极端跳跃)
        fomc_long = fomc_z > 2.5
        fomc_short = fomc_z < -2.5

        # Pulse 2: Bull Steepening Breakout (市场抢跑降息)
        # dgs2急剧下行(Z<-2.0) + 曲线变陡(Z>1.5) + 短端确认仍在下行
        bull_steep_long = (dgs2_z < -2.0) & (t10y2y_z > 1.5) & dgs2_falling

        # Pulse 3: Hawkish Extreme Exhaustion (Anti-Catch-Falling-Knife 完美示范)
        # dgs2极端飙升引发美债大跌(Z>2.0) + 曲线极度倒挂(Z<-1.5) + 飙升动能终结跌破3日均线(衰竭确认) = 抄底美债
        hawkish_exhaust_long = (dgs2_z > 2.0) & (t10y2y_z < -1.5) & dgs2_falling

        # Pulse 4: Bear Flattening Breakout (市场抢跑加息)
        bear_flat_short = (dgs2_z > 2.0) & (t10y2y_z < -1.5) & dgs2_rising

        # Pulse 5: Dovish Extreme Exhaustion (降息定价极度拥挤后反抽)
        dovish_exhaust_short = (dgs2_z < -2.0) & (t10y2y_z > 1.5) & dgs2_rising

        # --- Trigger Aggregation ---
        long_condition = fomc_long | bull_steep_long | hawkish_exhaust_long
        short_condition = fomc_short | bear_flat_short | dovish_exhaust_short

        signal[long_condition] = 1.0
        signal[short_condition] = -1.0

        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(momentum_window={self.momentum_window}, zscore_window={self.zscore_window})"