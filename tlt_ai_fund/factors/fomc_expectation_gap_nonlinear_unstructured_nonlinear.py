import numpy as np
import pandas as pd

class FomcExpectationGapNonlinearFactor:
    """Fomc Expectation Gap Nonlinear (unstructured/nonlinear)

    逻辑: 结合 NLP 提取的 FOMC 情绪得分极端突变与 VIX 状态交叉验证。在 VIX 高企且流动性紧张时，美联储的极端鸽派转向(Z>2.5)被视为有效救市信号；而在 VIX 较低且市场自满时，极端鹰派转向(Z<-2.5)将产生严重未 Price-in 的紧缩冲击。
    数据: fomc_sentiment, vixcls
    输出: 脉冲信号, 鸽派救市 +1.0 (看多美债), 鹰派紧缩 -1.0 (看空美债), 非触发日 0.0
    """

    def __init__(self, z_threshold: float = 2.5, vix_threshold: float = 20.0, rolling_window: int = 252):
        self.name = 'fomc_expectation_gap_nonlinear'
        self.z_threshold = z_threshold
        self.vix_threshold = vix_threshold
        self.rolling_window = rolling_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'fomc_sentiment' not in data.columns or 'vixcls' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        fomc = data['fomc_sentiment'].ffill()
        vix = data['vixcls'].ffill()
        
        # 只在 FOMC 情绪得分发生变动的交易日触发，确保脉冲型特征
        is_event_day = fomc.diff() != 0
        is_event_day = is_event_day.fillna(False)
        
        # 使用 t-1 的数据计算基准滚动均值与标准差，避免发生突变当天的数据扭曲偏离度
        fomc_shifted = fomc.shift(1)
        rolling_mean = fomc_shifted.rolling(window=self.rolling_window, min_periods=21).mean()
        rolling_std = fomc_shifted.rolling(window=self.rolling_window, min_periods=21).std()
        
        # 计算针对历史基准的 Z-Score，平滑极小波动以避免除以零假阳性
        z_score = (fomc - rolling_mean) / (rolling_std + 1e-4)
        
        signal = pd.Series(0.0, index=data.index, name=self.name)
        
        # 鸽派突变：FOMC 得分发生 +2.5Sigma 级别飙升，且市场处于高压恐慌环境 (VIX > 20)
        # 此时超预期的放水信号将带来长端利率的大幅下行
        bull_condition = is_event_day & (z_score > self.z_threshold) & (vix > self.vix_threshold)
        
        # 鹰派突变：FOMC 得分发生 -2.5Sigma 级别骤降，且市场尚未 Price-in 风险 (VIX <= 20)
        # 此时突然的紧缩信号将大幅拉升利率中枢
        bear_condition = is_event_day & (z_score < -self.z_threshold) & (vix <= self.vix_threshold)
        
        signal.loc[bull_condition] = 1.0
        signal.loc[bear_condition] = -1.0
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_threshold={self.z_threshold}, vix_threshold={self.vix_threshold}, rolling_window={self.rolling_window})"