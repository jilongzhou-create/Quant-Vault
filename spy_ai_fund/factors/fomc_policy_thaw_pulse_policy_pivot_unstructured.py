import numpy as np
import pandas as pd

class FomcPolicyThawPulseFactor:
    """Fomc Policy Thaw Pulse (policy_pivot/unstructured)

    逻辑: 捕捉美联储货币政策态度的边际解冻与收紧。当先前的 FOMC 情绪为鹰派(负值)且本次声明显著向鸽派偏移(跃升>0.1)时，市场预期反转，触发看多脉冲；反之鸽转鹰时触发看空。
    数据: fomc_sentiment
    输出: 1.0(看多，鹰派边际退潮), -1.0(看空，鸽派边际收缩), 0.0(常态休眠)
    触发条件: 前次情绪<0且跃升>=0.1看多；前次情绪>0且下挫<=-0.1看空。脉冲信号将维持5个交易日(资金重新配置的短窗口)，预期Trigger Rate在8%~12%之间。
    """

    def __init__(self, pulse_duration: int = 5, jump_threshold: float = 0.1):
        self.name = 'fomc_policy_thaw_pulse'
        # pulse_duration=5: 情绪冲击在美股通常发酵1周(5个交易日)
        self.pulse_duration = pulse_duration
        # jump_threshold=0.1: 范围在[-1, 1]的情绪打分中，0.1意味着5%跨度的实质性语句转向，滤除微小修辞变化
        self.jump_threshold = jump_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        # 确保数据连续性，处理空值
        sentiment = data['fomc_sentiment'].ffill()
        
        # 【边际变化铁律】: 只捕捉发生变化的瞬间，不使用绝对水位
        # 因为非会议日数据是前向填充的，所以 delta 平时必然为 0，只在会议发生次日存在非零跳跃
        delta = sentiment.diff()
        
        # 记录跳跃前的原有状态，用于判断"预期转向"
        prev_sentiment = sentiment.shift(1)
        
        raw_signal = pd.Series(0.0, index=data.index)
        
        # 看多脉冲: 之前处于鹰派状态(政策偏紧)，本次声明出现边际解冻(显著转鸽)
        long_cond = (prev_sentiment < 0.0) & (delta >= self.jump_threshold)
        
        # 看空脉冲: 之前处于鸽派状态(政策偏松)，本次声明出现边际收缩(显著转鹰)
        short_cond = (prev_sentiment > 0.0) & (delta <= -self.jump_threshold)
        
        raw_signal.loc[long_cond] = 1.0
        raw_signal.loc[short_cond] = -1.0
        
        # 扩展极短的脉冲窗口，使得每次事件维持数天，以满足目标 Trigger Rate
        signal = raw_signal.replace(0.0, np.nan).ffill(limit=self.pulse_duration - 1).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(pulse_duration={self.pulse_duration}, jump_threshold={self.jump_threshold})"