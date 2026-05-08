import numpy as np
import pandas as pd

class FomcSentimentPivotPulseFactor:
    """FOMC Sentiment Pivot Pulse (policy_pivot/unstructured)

    逻辑: 捕捉美联储FOMC声明情绪得分的边际剧变或趋势反转点(由鹰转鸽/由鸽转鹰)。市场定价对流动性拐点的反应通常集中在预期发生变化的极短窗口。
    数据: [fomc_sentiment]
    输出: +1.0代表鸽派突变(看多美股), -1.0代表鹰派突变(看空美股)。常态返回0.0。
    触发条件: 情绪得分日环比跳跃 > 0.25 或 跨越0轴产生反转。脉冲维持3个交易日，预期 Trigger Rate 约 8%-12%。
    """

    def __init__(self, jump_threshold: float = 0.25, pulse_window: int = 3):
        self.name = 'fomc_sentiment_pivot_pulse'
        # 0.25代表[-1.0, 1.0]区间内1/8的幅度跃升, 具有实质性政策指引变化的经济学含义
        self.jump_threshold = jump_threshold
        # 3天的维持窗口用于覆盖市场完全消化FOMC声明带来的定价重估期, 同时保障满足5-15%的触发率法则
        self.pulse_window = pulse_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        if 'fomc_sentiment' not in data.columns:
            return signal

        fomc = data['fomc_sentiment'].ffill()
        fomc_diff = fomc.diff()
        
        # 鸽派突变 (边际超预期鸽派 或 越过0轴由鹰派转向鸽派)
        dovish_trigger = (fomc_diff > self.jump_threshold) | ((fomc > 0.0) & (fomc.shift(1) < 0.0))
        
        # 鹰派突变 (边际超预期鹰派 或 越过0轴由鸽派转向鹰派)
        hawkish_trigger = (fomc_diff < -self.jump_threshold) | ((fomc < 0.0) & (fomc.shift(1) > 0.0))

        # 延展脉冲(动量延续期)，生成短暂的开仓窗口
        dovish_pulse = dovish_trigger.rolling(window=self.pulse_window, min_periods=1).max() > 0
        hawkish_pulse = hawkish_trigger.rolling(window=self.pulse_window, min_periods=1).max() > 0

        signal.loc[dovish_pulse] = 1.0
        signal.loc[hawkish_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(jump_threshold={self.jump_threshold}, pulse_window={self.pulse_window})"