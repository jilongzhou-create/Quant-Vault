import numpy as np
import pandas as pd

class FomcPolicyPivotPulseFactor:
    """政策转向情绪脉冲因子 (policy_pivot/unstructured)

    逻辑: 捕捉美联储FOMC声明鸽鹰情绪发生剧烈突变的瞬间。关注它的边际动量变化而非绝对水位。只有在情绪得分跳跃时才确认政策预期发生了改变，从而在短期内引发流动性冲量和重定价。
    数据: [fomc_sentiment]
    输出: 鸽派突变看多(+1.0), 鹰派突变看空(-1.0), 常态休眠(0.0)
    触发条件: 情绪得分日度单次边际跳跃 >= 0.20 或 <= -0.20, 在随后7天的极短定价窗口期内维持信号。预期 Trigger Rate 在 5%-15% 之间。
    """

    def __init__(self, jump_threshold: float = 0.20, window_days: int = 7):
        self.name = 'fomc_policy_pivot_pulse'
        # 阈值 0.20 代表 FOMC 措辞至少出现 20% 幅度的方向性偏移，为实质性政策反转
        self.jump_threshold = jump_threshold
        # 窗口期 7 天代表市场对宏观政策转向完成重新定价的极短缓冲期 (约1.5周)
        self.window_days = window_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 非频日前向填充
        fomc_sen = data['fomc_sentiment'].ffill()
        
        # 【边际变化铁律】: 绝对禁止直接判断绝对值，由于是低频阶梯数据，必须使用 diff() 捕捉公布当天的变化脉冲
        sen_diff = fomc_sen.diff()
        
        # 捕捉剧烈的鸽派突变和鹰派突变瞬间
        bull_jump = sen_diff >= self.jump_threshold
        bear_jump = sen_diff <= -self.jump_threshold
        
        # 将突变信号延续极短的几日，以便捕获市场的交易冲量
        bull_pulse = bull_jump.rolling(window=self.window_days, min_periods=1).max() > 0
        bear_pulse = bear_jump.rolling(window=self.window_days, min_periods=1).max() > 0
        
        # 【零值休眠铁律】
        signal = pd.Series(0.0, index=data.index)
        signal.loc[bull_pulse] = 1.0
        signal.loc[bear_pulse] = -1.0
        
        # 冲突防御
        conflict = bull_pulse & bear_pulse
        signal.loc[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(jump_threshold={self.jump_threshold}, window_days={self.window_days})"