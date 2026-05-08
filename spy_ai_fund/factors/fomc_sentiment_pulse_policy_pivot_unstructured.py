import numpy as np
import pandas as pd

class FomcSentimentPulseFactor:
    """FomcSentimentPulseFactor (policy_pivot/unstructured)

    逻辑: 捕捉美联储FOMC声明情绪的边际剧变(鸽派突变看多, 鹰派突变看空), 并在随后的极短窗口内保持脉冲输出, 给予市场消化期。
    数据: [fomc_sentiment]
    输出: 鸽派转向(预期放宽)输出+1.0, 鹰派突变(预期收紧)输出-1.0
    触发条件: 情绪得分单日跳跃>0.25, 或完成鹰鸽极性反转。信号保持4天, 预期Trigger Rate ~6%-10%。
    """

    def __init__(self, diff_threshold: float = 0.25, hold_days: int = 4):
        self.name = 'fomc_sentiment_pulse'
        self.diff_threshold = diff_threshold
        self.hold_days = hold_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            signal.name = self.name
            return signal

        fomc = data['fomc_sentiment'].ffill()
        fomc_diff = fomc.diff()
        prev_fomc = fomc.shift(1)
        
        # 边际变化铁律: 极性反转判断 (前值在显著鹰派/鸽派区间, 当前值跨越中性区进入对立面)
        dovish_reversal = (prev_fomc < -0.1) & (fomc > 0.1)
        hawkish_reversal = (prev_fomc > 0.1) & (fomc < -0.1)
        
        # 动量剧变: 情绪得分的边际跳跃幅度超过阈值
        dovish_jump = (fomc_diff > self.diff_threshold) | dovish_reversal
        hawkish_jump = (fomc_diff < -self.diff_threshold) | hawkish_reversal
        
        # 脉冲生成: 保持hold_days天以达到5%-15%的Trigger Rate, 同时符合物理上的突发事件消化期
        dovish_pulse = dovish_jump.astype(float).rolling(window=self.hold_days, min_periods=1).max()
        hawkish_pulse = hawkish_jump.astype(float).rolling(window=self.hold_days, min_periods=1).max()
        
        # 合成信号
        raw_signal = dovish_pulse - hawkish_pulse
        
        # 覆盖到默认信号中
        signal.update(raw_signal)
        signal = signal.fillna(0.0).clip(-1.0, 1.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(diff_threshold={self.diff_threshold}, hold_days={self.hold_days})"