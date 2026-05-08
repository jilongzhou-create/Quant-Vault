import numpy as np
import pandas as pd

class FomcHawkDoveReversalFactor:
    """FomcHawkDoveReversal (unstructured/unstructured)

    逻辑: 捕捉 FOMC 会议引发的极端宏观政策预期反转。仅当底层 NLP 情绪得分发生剧烈跳变并彻底跨越零轴时提取转折信号。动量不再扩大的次日（衰竭确认）才触发，完美规避会议当天的极度无序波动。
    数据: fomc_sentiment
    触发: 5日变化量的252日 Z-Score > 2.5 且由负(鹰)转正(鸽)，同时单日动量增量回落 (diff <= 0 衰竭确认) 触发看多。
    输出: 狙击级脉冲，鹰转鸽看多美债(+1.0)，鸽转鹰看空美债(-1.0)。
    """

    def __init__(self, window=252, mom_window=5):
        self.name = 'fomc_hawk_dove_reversal'
        self.window = window
        self.mom_window = mom_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # 前向填充阶梯数据，防 NaN
        sentiment = data['fomc_sentiment'].ffill().fillna(0)
        
        # 铁律3: 边际变化 (提取预期突变瞬间的跃升量)
        mom = sentiment.diff(self.mom_window)
        
        # 计算 Z-Score 捕捉极端事件
        roll_mean = mom.rolling(self.window, min_periods=self.window//2).mean()
        roll_std = mom.rolling(self.window, min_periods=self.window//2).std()
        
        zscore = pd.Series(0.0, index=mom.index)
        valid = roll_std > 1e-6
        zscore[valid] = (mom[valid] - roll_mean[valid]) / roll_std[valid]
        
        # 铁律2: 二阶导数 (衰竭确认)
        # 对于阶梯跳跃数据，事件日动量暴增，次日动量不再扩大，此即天然的"指标极端 + 开始回落/衰竭"
        mom_diff = mom.diff(1)
        
        # 寻找情绪底色大逆转 (Hawkish < 0, Dovish > 0)
        was_hawk = sentiment.shift(self.mom_window) < 0
        is_dove = sentiment > 0
        
        was_dove = sentiment.shift(self.mom_window) > 0
        is_hawk = sentiment < 0
        
        # 鹰派枯竭，向鸽派极端反转 (看多美债)
        long_cond = (
            (zscore > 2.5) & 
            (was_hawk & is_dove) & 
            (mom_diff <= 0)
        )
        
        # 鸽派枯竭，向鹰派极端反转 (看空美债)
        short_cond = (
            (zscore < -2.5) & 
            (was_dove & is_hawk) & 
            (mom_diff >= 0)
        )
        
        # 铁律1: 零值休眠 (Sniper Pulse 只在跨过条件的瞬间发出一发子弹)
        long_pulse = long_cond & (~long_cond.shift(1).fillna(False))
        short_pulse = short_cond & (~short_cond.shift(1).fillna(False))
        
        signal[long_pulse] = 1.0
        signal[short_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, mom_window={self.mom_window})"