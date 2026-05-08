import numpy as np
import pandas as pd

class FomcSentimentContrarianPivotFactor:
    """Fomc Sentiment Contrarian Pivot (policy_pivot / unstructured)

    逻辑: 捕捉美联储在特定政策周期下的预期反转。在相对鹰派的环境下(前期情绪偏低), 若FOMC声明措辞突然出现显著的边际鸽派改善, 说明紧缩周期濒临崩溃, 构成左侧抄底看多脉冲; 反之, 在宽松周期中突然变鹰, 预示流动性红利见顶, 触发看空脉冲。
    数据: [fomc_sentiment]
    输出: 1.0 (反转鸽化看多), -1.0 (反转鹰化看空), 0.0 (平稳期或无显著变动)
    触发条件: 声明情绪发生 > 0.08 的跳变, 且前期环境并非已处于同向极值区。触发当天及随后5个交易日内持续输出脉冲。预期 Trigger Rate: 8% 左右。
    """

    def __init__(self):
        self.name = 'fomc_sentiment_contrarian_pivot'
        self.jump_threshold = 0.08  # 显著的态度跳跃阈值
        self.regime_limit = 0.1     # 环境限制, 防止过度顺势
        self.pulse_window = 6       # T + 5 (共6天脉冲), 维持 Trigger Rate 5%-15%

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        sentiment = data['fomc_sentiment'].ffill()
        
        # 严格遵守边际变化铁律, 绝对禁止使用绝对值直接生成信号
        diff_sentiment = sentiment.diff().fillna(0.0)
        prev_sentiment = sentiment.shift(1).fillna(0.0)
        
        # 突变鸽派: 边际变化大于阈值, 且发生在此前并非极度鸽派的环境下
        bull_pivot = (diff_sentiment >= self.jump_threshold) & (prev_sentiment <= self.regime_limit)
        
        # 突变鹰派: 边际变化小于负阈值, 且发生在此前并非极度鹰派的环境下
        bear_pivot = (diff_sentiment <= -self.jump_threshold) & (prev_sentiment >= -self.regime_limit)
        
        # 展期形成极短周期的脉冲 (6个交易日)
        bull_pulse = bull_pivot.astype(float).rolling(window=self.pulse_window, min_periods=1).max()
        bear_pulse = bear_pivot.astype(float).rolling(window=self.pulse_window, min_periods=1).max()
        
        # 合成信号
        signal = bull_pulse - bear_pulse
        
        # 确保数据纯净且限制在 [-1.0, 1.0] 范围内
        signal = signal.fillna(0.0).clip(-1.0, 1.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(jump_threshold={self.jump_threshold}, pulse_window={self.pulse_window})"