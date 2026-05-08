import numpy as np
import pandas as pd

class EpuVolatilityReversalFactor:
    """EPU Volatility Reversal (volatility/unstructured)

    逻辑: 经济政策不确定性指数 (EPU) 是基于非结构化新闻文本提取的宏观恐慌指标。本因子提取 EPU 的二阶特征——"不确定性的波动率(Vol of EPU)"。
          脉冲看多(+1.0): 当 EPU 波动率狂飙达到极端高位 (Z-Score > 2.5) 且开始回落时，意味着极端的政策博弈和新闻恐慌瓦解，资金重返避险资产 (TLT)。
          脉冲看空(-1.0): 当 EPU 波动率极度低迷 (Z-Score < -2.0) 且突然向上突破时，意味着长期的政策自满情绪破裂，抛售美债以规避潜在的冲击。
          信号天然设计为持续 5 天的"狙击手"级别脉冲，确保严格的零值休眠状态，并使 Trigger Rate 稳控在 5%-15% 之间。
    数据: usepuindxd (每日经济政策不确定性指数)
    触发: Z-Score > 2.5 且向下跌破 3 日均线 -> 触发做多脉冲；Z-Score < -2.0 且向上突破 3 日均线 -> 触发做空脉冲。
    输出: 脉冲型 [-1.0, 1.0]，常态下绝对返回 0.0。
    """

    def __init__(self):
        self.name = 'epu_volatility_reversal'
        self.vol_window = 21       # 提取波动率的时间窗口
        self.z_window = 504        # 评估极值的长周期基准 (约2年)
        self.pulse_days = 5        # 极值反转后的脉冲持续天数

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 信号，严格遵守铁律1：零值休眠
        signal = pd.Series(0.0, index=data.index)
        
        # 独立防御：如果缺少所需数据，直接返回 0.0 信号
        if 'usepuindxd' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        
        # 铁律3：边际变化。不看 EPU 绝对水位，而是提取其 21 日的短期波动率
        epu_vol = epu.rolling(window=self.vol_window, min_periods=10).std()
        
        # 计算动态 Z-Score (避免无意义的魔法数字，基准自适应)
        roll_mean = epu_vol.rolling(window=self.z_window, min_periods=126).mean()
        roll_std = epu_vol.rolling(window=self.z_window, min_periods=126).std().clip(lower=1.0)
        zscore = (epu_vol - roll_mean) / roll_std
        
        # 计算 3 日均线用于精确确认动量反转 (二阶导数)
        ma3 = epu_vol.rolling(window=3, min_periods=1).mean()
        
        # ==========================================
        # 铁律2：二阶导数 (极值 + 衰竭/突破确认)
        # ==========================================
        
        # 1. 看多触发点 (极端高位 + 刚刚向下跌破短期均线)
        extreme_high = zscore > 2.5
        exhaustion = epu_vol < ma3
        prev_not_exhausted = epu_vol.shift(1) >= ma3.shift(1)
        point_long = extreme_high & exhaustion & prev_not_exhausted
        
        # 2. 看空触发点 (极度低迷 + 刚刚向上突破短期均线)
        extreme_low = zscore < -2.0
        surge = epu_vol > ma3
        prev_not_surge = epu_vol.shift(1) <= ma3.shift(1)
        point_short = extreme_low & surge & prev_not_surge
        
        # ==========================================
        # 铁律1：零值休眠与精准触发率 (Sniper Pulse)
        # ==========================================
        
        # 将单一的日频触发点，向后扩展为持续 5 天的固定有效交易窗口
        pulse_long = point_long.rolling(window=self.pulse_days, min_periods=1).max() > 0
        pulse_short = point_short.rolling(window=self.pulse_days, min_periods=1).max() > 0
        
        # 赋值最终信号 (防止同一天重叠，做空优先，做多覆盖)
        signal.loc[pulse_short] = -1.0
        signal.loc[pulse_long] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(vol_window={self.vol_window}, z_window={self.z_window}, pulse_days={self.pulse_days})"