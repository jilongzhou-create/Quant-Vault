import numpy as np
import pandas as pd

class FomcSentimentMarginalPulseFactor:
    """FomcSentimentMarginalPulseFactor (policy_pivot/unstructured)

    逻辑: 捕捉FOMC声明情绪的超预期边际突变(鹰转鸽或鸽转鹰)。市场不在乎绝对鸽/鹰，只在乎超预期的边际改变。一旦发生超过特定阈值的情绪跃升或由负转正的结构性反转，在随后的一周(5个交易日)内输出强烈的方向脉冲，发酵政策转向预期。
    数据: fomc_sentiment
    输出: 鸽派突变产生+1.0(看多)，鹰派突变产生-1.0(看空)，常态为0.0
    触发条件: 情绪单次大幅跳升>=0.2，或发生明显的正负极性反转。信号持续5个交易日，预期Trigger Rate控制在5%-15%。
    """

    def __init__(self, jump_threshold=0.20, reversal_threshold=0.05, hold_days=5):
        self.name = 'fomc_sentiment_marginal_pulse'
        self.jump_threshold = jump_threshold
        self.reversal_threshold = reversal_threshold
        self.hold_days = hold_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        # 确保无空值，且采用前向填充维持低频数据的阶梯态
        sentiment = data['fomc_sentiment'].ffill()
        sentiment_diff = sentiment.diff()
        
        # 鸽派突变：大幅向鸽派跃升，或者由偏鹰转为偏鸽(跨越中轴反转)
        dovish_jump = sentiment_diff >= self.jump_threshold
        dovish_reversal = (sentiment.shift(1) <= -self.reversal_threshold) & \
                          (sentiment >= self.reversal_threshold) & \
                          (sentiment_diff > 0)
        dovish_trigger = dovish_jump | dovish_reversal
        
        # 鹰派突变：大幅向鹰派下挫，或者由偏鸽转为偏鹰(跨越中轴反转)
        hawkish_jump = sentiment_diff <= -self.jump_threshold
        hawkish_reversal = (sentiment.shift(1) >= self.reversal_threshold) & \
                           (sentiment <= -self.reversal_threshold) & \
                           (sentiment_diff < 0)
        hawkish_trigger = hawkish_jump | hawkish_reversal
        
        # 将脉冲向后发酵 hold_days 天（利用 rolling max 代表这几天内曾有过触发）
        dovish_发酵 = dovish_trigger.rolling(window=self.hold_days, min_periods=1).max().fillna(0).astype(bool)
        hawkish_发酵 = hawkish_trigger.rolling(window=self.hold_days, min_periods=1).max().fillna(0).astype(bool)
        
        signal = pd.Series(0.0, index=data.index)
        
        # 极低概率下如果多空同时触发（实际逻辑上互斥），设为 0
        signal[hawkish_发酵] = -1.0
        signal[dovish_发酵] = 1.0
        signal[dovish_发酵 & hawkish_发酵] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(jump_threshold={self.jump_threshold}, hold_days={self.hold_days})"