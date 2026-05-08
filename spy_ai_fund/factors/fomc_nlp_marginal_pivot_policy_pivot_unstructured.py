import numpy as np
import pandas as pd

class FomcNlpMarginalPivotFactor:
    """FOMC非结构化情绪突变因子 (policy_pivot/unstructured)

    逻辑: 捕捉美联储货币政策声明中情绪预期的瞬间反转。绝对禁止使用低频文本得分的绝对水位，只关注其边际动量跳跃。当文本情绪发生剧烈鸽派突变(>0.3)或由鹰转鸽时触发多头脉冲；反之触发空头脉冲。
    数据: fomc_sentiment
    输出: 1.0 (鸽派情绪突变/看多美股), -1.0 (鹰派情绪突变/看空美股), 0.0 (常态休眠)
    触发条件: 情绪得分发生 >0.3 的阶跃或跨零轴反转触发多头，<-0.3 的坠跌或跨零轴反转触发空头，信号维持 4 个交易日，以达成 5%-15% 的目标 Trigger Rate
    """

    def __init__(self, jump_threshold=0.3, hold_days=4):
        self.name = 'fomc_nlp_marginal_pivot'
        self.jump_threshold = jump_threshold
        self.hold_days = hold_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 必须处理数据缺失情况，默认返回全 0 的休眠 Series
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        # 前向填充非会议日的数据，还原出阶梯状的低频真实序列
        fomc = data['fomc_sentiment'].ffill()
        
        # 严格遵守边际变化铁律：只看预期改变的瞬间
        fomc_diff = fomc.diff().fillna(0.0)
        
        # 1. 鸽派突变：情绪得分骤升超过阈值，或者从鹰派(负)直接跨零轴强力反转为鸽派(正)
        dovish_jump = fomc_diff > self.jump_threshold
        dovish_turn = (fomc.shift(1) < 0.0) & (fomc > 0.0)
        dovish_trigger = dovish_jump | dovish_turn
        
        # 2. 鹰派突变：情绪得分骤降超过阈值，或者从鸽派(正)直接跨零轴强力反转为鹰派(负)
        hawkish_jump = fomc_diff < -self.jump_threshold
        hawkish_turn = (fomc.shift(1) > 0.0) & (fomc < 0.0)
        hawkish_trigger = hawkish_jump | hawkish_turn
        
        # 3. 零值休眠铁律：突变发生当天及随后极短的 hold_days 内输出脉冲信号
        # rolling.max() 在此处不会造成未来函数，因为它是对历史窗口内的 trigger 进行判定
        bull_pulse = dovish_trigger.rolling(window=self.hold_days, min_periods=1).max() > 0
        bear_pulse = hawkish_trigger.rolling(window=self.hold_days, min_periods=1).max() > 0
        
        signal = pd.Series(0.0, index=data.index)
        signal[bull_pulse] = 1.0
        signal[bear_pulse] = -1.0
        
        # 消除重叠期可能导致的逻辑冲突
        conflict = bull_pulse & bear_pulse
        signal[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(jump_threshold={self.jump_threshold}, hold_days={self.hold_days})"