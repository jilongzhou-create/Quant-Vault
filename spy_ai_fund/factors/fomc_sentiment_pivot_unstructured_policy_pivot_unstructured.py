import numpy as np
import pandas as pd

class FomcSentimentPivotUnstructuredFactor:
    """FOMC情绪边际突变脉冲因子 (policy_pivot/unstructured)

    逻辑: 捕捉美联储FOMC声明情绪的剧变时刻(NLP得分突变或穿越零轴)。根据边际变化铁律，阶梯状数据绝对禁止使用绝对值。当美联储情绪发生鹰鸽转向的瞬间，市场在极短几日内会对流动性冲量重新定价，此时进行顺势狙击。
    数据: [fomc_sentiment]
    输出: 鸽派突变产生+1.0(看多美股)脉冲; 鹰派突变产生-1.0(看空美股)脉冲
    触发条件: 情绪跳跃幅度 >= 0.2 或 穿越零轴产生反转。脉冲持续3个交易日, 预期 Trigger Rate 控制在 5%-10% 左右。
    """

    def __init__(self, jump_threshold=0.2, pulse_window=3):
        self.name = 'fomc_sentiment_pivot_unstructured'
        self.jump_threshold = jump_threshold
        self.pulse_window = pulse_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # fomc_sentiment 是阶梯状的低频宏观数据, 需要前向填充以维持最新状态
        sentiment = data['fomc_sentiment'].ffill()
        
        # 绝对铁律: 必须使用边际变化(.diff)捕捉预期改变的瞬间
        sentiment_diff = sentiment.diff()
        prev_sentiment = sentiment.shift(1)
        
        if sentiment_diff.isna().all():
            return signal

        # 鸽派转向突变 (流动性改善预期 -> 强看多美股)
        # 条件1: 情绪向鸽派方向产生了超过阈值的巨大跳跃
        # 条件2: 情绪由鹰派(负得分)转变为鸽派(正得分)的定性反转
        dovish_jump = sentiment_diff >= self.jump_threshold
        dovish_cross = (prev_sentiment < 0) & (sentiment >= 0) & (sentiment_diff > 0)
        dovish_trigger = dovish_jump | dovish_cross
        
        # 鹰派转向突变 (流动性收紧预期 -> 强看空美股)
        # 条件1: 情绪向鹰派方向产生了超过阈值的巨大跳跃
        # 条件2: 情绪由鸽派(正得分)转变为鹰派(负得分)的定性反转
        hawkish_jump = sentiment_diff <= -self.jump_threshold
        hawkish_cross = (prev_sentiment > 0) & (sentiment <= 0) & (sentiment_diff < 0)
        hawkish_trigger = hawkish_jump | hawkish_cross
        
        # 将极端脉冲信号向后延续极短的几天, 以捕捉市场重新定价的窗口
        # 使用 rolling.max() 将单日的 True(1) 延续到 pulse_window 天
        dovish_pulse = dovish_trigger.rolling(window=self.pulse_window, min_periods=1).max()
        hawkish_pulse = hawkish_trigger.rolling(window=self.pulse_window, min_periods=1).max()
        
        # 生成狙击手级别的脉冲信号
        signal[dovish_pulse == 1] = 1.0
        signal[hawkish_pulse == 1] = -1.0
        
        # 消除极小概率下的双向冲突
        conflict = (dovish_pulse == 1) & (hawkish_pulse == 1)
        signal[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(jump_threshold={self.jump_threshold}, pulse_window={self.pulse_window})"