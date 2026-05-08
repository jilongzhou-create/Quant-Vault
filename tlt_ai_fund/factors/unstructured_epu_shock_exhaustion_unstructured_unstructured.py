import numpy as np
import pandas as pd

class UnstructuredEpuShockExhaustionFactor:
    """经济政策不确定性突变衰竭因子 (Unstructured/Sentiment)

    逻辑: EPU(Economic Policy Uncertainty)基于新闻文本挖掘, 反映宏观政策和经济的混沌程度. 当不确定性在一个月内发生极端飙升时, 往往对应突发黑天鹅或政策剧变, 催生避险需求. 但恐慌主跌浪中可能伴随流动性无差别抛售(如2020年3月), 必须等不确定性边际回落瞬间, 才能确认恐慌见顶, 输出看多美债的脉冲.
    数据: usepuindxd (Economic Policy Uncertainty Index for US)
    触发: 不确定性21日变化量的Z-Score > 2.5(极端突变) + 短期动量反转回落(衰竭确认).
    输出: 脉冲型, +1.0 看多美债(不确定性飙升见顶衰竭), -1.0 看空美债(不确定性骤降极度自满后反弹).
    """

    def __init__(self, window=21, z_window=252, smooth_window=5):
        self.name = 'unstructured_epu_shock_exhaustion'
        self.window = window
        self.z_window = z_window
        self.smooth_window = smooth_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 提取数据并前向填充缺失值
        epu = data['usepuindxd'].ffill()
        
        # 铁律3: 边际变化 Only
        # 先对日频噪音极大的 EPU 进行平滑, 然后计算 1 个月(21日)的边际突变动量
        epu_smooth = epu.rolling(window=self.smooth_window).mean()
        epu_diff = epu_smooth.diff(self.window)
        
        # 极端突变: 计算滚动 252 日(1年)的 Z-Score
        epu_diff_mean = epu_diff.rolling(window=self.z_window).mean()
        epu_diff_std = epu_diff.rolling(window=self.z_window).std()
        
        epu_diff_z = pd.Series(0.0, index=data.index)
        valid_idx = epu_diff_std > 1e-6
        epu_diff_z[valid_idx] = (epu_diff[valid_idx] - epu_diff_mean[valid_idx]) / epu_diff_std[valid_idx]
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 必须同时满足极值飙升 + 开始边际回落
        # 看多衰竭: 不确定性变化量升速放缓(跌破5日均值) 且 短期走势(3日)实质性回落
        momentum_fading_long = (epu_diff < epu_diff.rolling(window=5).mean()) & (epu_smooth.diff(3) < 0)
        
        # 看空衰竭: 不确定性骤降(极度自满)动能放缓 且 短期走势(3日)开始回升反弹
        momentum_fading_short = (epu_diff > epu_diff.rolling(window=5).mean()) & (epu_smooth.diff(3) > 0)
        
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 仅在同时满足极端 Z-Score 和衰竭二阶导条件时触发脉冲
        cond_long = (epu_diff_z > 2.5) & momentum_fading_long
        cond_short = (epu_diff_z < -2.5) & momentum_fading_short
        
        signal[cond_long] = 1.0
        signal[cond_short] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_window={self.z_window}, smooth_window={self.smooth_window})"