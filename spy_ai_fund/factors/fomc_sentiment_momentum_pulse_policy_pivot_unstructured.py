import numpy as np
import pandas as pd

class FomcSentimentMomentumPulseFactor:
    """FomcSentimentMomentumPulse (policy_pivot/unstructured)

    逻辑: 捕捉美联储FOMC声明情绪评分的非线性阶梯跳跃(鸽派突变看多, 鹰派突变看空), 仅在市场流动性预期反转的瞬间及随后极短几天内释放交易冲量。
    数据: [fomc_sentiment]
    输出: 鸽派情绪突发跃升(看多流动性)输出+1.0脉冲, 鹰派情绪突发骤降(看空流动性)输出-1.0脉冲, 常态0.0。
    触发条件: 情绪得分单日跳跃幅度>0.15, 或发生鹰鸽轴反转(零轴穿越)且跳跃>0.1, 脉冲持续3天。预期Trigger Rate 6%-10%。
    """

    def __init__(self, jump_threshold: float = 0.15, zero_cross_threshold: float = 0.10, pulse_window: int = 3):
        self.name = 'fomc_sentiment_momentum_pulse'
        # 0.15 代表 FOMC 情绪得分上发生了 15% 的显著预期重估
        self.jump_threshold = jump_threshold
        # 0.10 代表 鹰鸽转向时所要求的最低重估动量
        self.zero_cross_threshold = zero_cross_threshold
        # 脉冲维持极短的时间窗口 (3个交易日), 消化完即陷入休眠
        self.pulse_window = pulse_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # FOMC数据为非会议日前向填充, 其变化率(diff)呈零星脉冲状
        sentiment = data['fomc_sentiment'].ffill()
        sentiment_diff = sentiment.diff()
        sentiment_prev = sentiment.shift(1)
        
        # 鸽派突变 (Bullish Liquidity): 情绪单日大幅上升，或由鹰(负)转鸽(正)且有确认幅度的上升
        dovish_jump = sentiment_diff > self.jump_threshold
        dovish_cross = (sentiment > 0.0) & (sentiment_prev < 0.0) & (sentiment_diff > self.zero_cross_threshold)
        dovish_event = dovish_jump | dovish_cross
        
        # 鹰派突变 (Bearish Liquidity): 情绪单日大幅下降，或由鸽(正)转鹰(负)且有确认幅度的下降
        hawkish_jump = sentiment_diff < -self.jump_threshold
        hawkish_cross = (sentiment < 0.0) & (sentiment_prev > 0.0) & (sentiment_diff < -self.zero_cross_threshold)
        hawkish_event = hawkish_jump | hawkish_cross
        
        # 转化为极短期的脉冲窗口 (向后延伸极短天数以捕捉趋势爆发期)
        dovish_pulse = dovish_event.rolling(window=self.pulse_window, min_periods=1).max().fillna(0).astype(bool)
        hawkish_pulse = hawkish_event.rolling(window=self.pulse_window, min_periods=1).max().fillna(0).astype(bool)
        
        # 注入信号，确保常态下返回0.0
        signal.loc[dovish_pulse] = 1.0
        signal.loc[hawkish_pulse] = -1.0
        
        # 极端情况下若鹰鸽同时触发则保持0.0 (实务中单向变动极难重叠)
        conflict = dovish_pulse & hawkish_pulse
        signal.loc[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(jump_threshold={self.jump_threshold}, zero_cross_threshold={self.zero_cross_threshold}, pulse_window={self.pulse_window})"