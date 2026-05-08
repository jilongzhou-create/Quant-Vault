import numpy as np
import pandas as pd

class EpuPanicExhaustionPulseFactor:
    """非结构化政策不确定性恐慌衰竭脉冲因子 (panic_mean_reversion/unstructured)

    逻辑: 极端的政策不确定性(EPU)代表宏观恐慌。均值回归市场中，当EPU创出高位极值并开始向下衰竭时，是极佳的抄底买点(恐慌出尽)；反之，当EPU从相对平静期突然加速上升，但尚未达到极值时，处于恐慌发酵的主跌浪，应看空。
    数据: usepuindxd (美国每日经济政策不确定性指数)
    输出: 1.0 (恐慌衰竭，看多), -1.0 (恐慌加速上升，看空), 0.0 (常态)
    触发条件: EPU Z-Score > 1.0 且 5日均线下穿15日均线看多；Z-Score在0至1.0间且5日内飙升0.8个标准差看空，预期 Trigger Rate 7-10%
    """

    def __init__(self):
        self.name = 'epu_panic_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        epu = data['usepuindxd'].ffill()
        
        # Calculate 252-day rolling stats for macro context
        roll_mean = epu.rolling(window=252, min_periods=63).mean()
        roll_std = epu.rolling(window=252, min_periods=63).std()
        
        z_score = (epu - roll_mean) / (roll_std + 1e-6)
        
        # Moving averages for momentum detection
        ma5 = epu.rolling(window=5).mean()
        ma15 = epu.rolling(window=15).mean()
        
        # Rate of change
        diff1 = epu.diff(1)
        diff5 = epu.diff(5)
        
        signal = pd.Series(0.0, index=data.index)
        
        # 极值 + 衰竭 = 强看多 (+1.0)
        # 条件：Z-Score > 1.0 (恐慌高位), 短期动量向下(ma5 < ma15)，且今日不确定性继续回落(diff1 < 0)
        bull_cond = (
            (z_score > 1.0) &
            (ma5 < ma15) &
            (diff1 < 0)
        )
        
        # 轻度/发酵期恐慌 + 加速爆发 = 强看空 (-1.0)
        # 条件：Z-Score 介于 0 和 1.0 之间（恐慌上升期，尚未极度恐慌），短期动量向上，且5日内出现显著跳升
        bear_cond = (
            (z_score > 0.0) & 
            (z_score <= 1.0) &
            (ma5 > ma15) &
            (diff5 > roll_std * 0.8) &
            (diff1 > 0)
        )
        
        signal[bull_cond] = 1.0
        signal[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"