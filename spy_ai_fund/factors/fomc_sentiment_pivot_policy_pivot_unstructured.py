import numpy as np
import pandas as pd

class FomcSentimentPivotFactor:
    """政策转向/非结构化数据 (policy_pivot/unstructured)

    逻辑: 捕捉FOMC声明的NLP情感得分(鹰鸽倾向)的剧烈边际变化。不要关注绝对情感是鹰是鸽，只关注瞬间的反转与跳跃。鸽派突变看多, 鹰派突变看空。
    数据: [fomc_sentiment]
    输出: +1.0表示鸽派突变(看多), -1.0表示鹰派突变(看空), 常态下返回0.0
    触发条件: 情感得分单次跳跃 > 0.25 或明显穿越零轴(情绪反转), 脉冲信号持续4个交易日(约两周内), 预期Trigger Rate约为10%-12%
    """

    def __init__(self, diff_threshold: float = 0.25, hold_days: int = 4):
        self.name = 'fomc_sentiment_pivot'
        self.diff_threshold = diff_threshold
        self.hold_days = hold_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index, name=self.name)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # FOMC情绪分数是低频阶梯状数据，向前填充后计算日度变化
        fomc = data['fomc_sentiment'].ffill()
        fomc_diff = fomc.diff()
        
        # 鸽派突变 (Bullish): 情绪分数剧烈上升，或者从明显偏鹰(-0.05)转为偏鸽(+0.05)
        dovish_jump = (fomc_diff >= self.diff_threshold) | ((fomc > 0.05) & (fomc.shift(1) < -0.05))
        
        # 鹰派突变 (Bearish): 情绪分数剧烈下降，或者从明显偏鸽(+0.05)转为偏鹰(-0.05)
        hawkish_jump = (fomc_diff <= -self.diff_threshold) | ((fomc < -0.05) & (fomc.shift(1) > 0.05))
        
        # 延长脉冲信号，以满足 5%-15% 的 Trigger Rate 铁律 (每年约8次会议 * 4天 = 约32个交易日触发 = ~12%)
        dovish_pulse = dovish_jump.rolling(window=self.hold_days, min_periods=1).max().fillna(0)
        hawkish_pulse = hawkish_jump.rolling(window=self.hold_days, min_periods=1).max().fillna(0)
        
        # 组装最终脉冲信号
        signal_array = np.where(dovish_pulse > 0, 1.0, 
                       np.where(hawkish_pulse > 0, -1.0, 0.0))
                       
        signal = pd.Series(signal_array, index=data.index, name=self.name)
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(diff_threshold={self.diff_threshold}, hold_days={self.hold_days})"