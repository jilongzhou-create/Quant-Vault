import numpy as np
import pandas as pd

class EpuShockReversalFactor:
    """经济政策不确定性突变衰竭因子 (microstructure/unstructured)

    逻辑: 经济政策不确定性(EPU)的短期极度飙升通常对应宏观恐慌和流动性冲击(类似2020年3月)。
          当EPU飙升至极值且在次日开始回落时, 标志着恐慌见顶和政策干预预期的兑现, 
          此时流动性危机解除, 避险资金大规模涌入美债, 触发做多TLT脉冲。反之亦然。
    数据: usepuindxd (Daily Economic Policy Uncertainty Index)
    触发: 3日变化量的252日 Z-Score > 2.5 且当日指标较前日下降 -> +1.0 (恐慌衰竭, 买入美债)
          3日变化量的252日 Z-Score < -2.5 且当日指标较前日上升 -> -1.0 (极度自满反转, 卖出美债)
    输出: 极短期脉冲信号 [-1.0, 0.0, 1.0]
    """

    def __init__(self, z_threshold: float = 2.5, diff_window: int = 3, rolling_window: int = 252):
        self.name = 'epu_shock_reversal'
        self.z_threshold = z_threshold
        self.diff_window = diff_window
        self.rolling_window = rolling_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 前向填充缺失值, 防止由于数据发布延迟导致的计算错误
        epu = data['usepuindxd'].ffill()
        
        # 铁律3: 边际变化 (只看政策不确定性的短期动量)
        epu_mom = epu.diff(self.diff_window)
        
        # 计算动量的滚动 Z-Score
        epu_mom_mean = epu_mom.rolling(window=self.rolling_window, min_periods=60).mean()
        epu_mom_std = epu_mom.rolling(window=self.rolling_window, min_periods=60).std()
        
        # 避免除以零
        epu_mom_z = (epu_mom - epu_mom_mean) / epu_mom_std.replace(0, np.nan)
        
        # 铁律2: 二阶导数 (衰竭条件)
        # 动量极高但当天指标已经开始回落
        is_exhausting_high = epu.diff(1) < 0
        
        # 动量极低但当天指标已经开始反弹
        is_exhausting_low = epu.diff(1) > 0
        
        # 铁律1: 零值休眠与狙击手脉冲触发
        long_trigger = (epu_mom_z > self.z_threshold) & is_exhausting_high
        short_trigger = (epu_mom_z < -self.z_threshold) & is_exhausting_low
        
        signal[long_trigger] = 1.0
        signal[short_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_threshold={self.z_threshold}, diff_window={self.diff_window}, rolling_window={self.rolling_window})"