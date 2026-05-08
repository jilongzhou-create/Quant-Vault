import numpy as np
import pandas as pd

class EpuExhaustionPulseFactor:
    """经济政策不确定性反转脉冲因子 (microstructure/unstructured)

    逻辑: 极高的政策不确定性往往伴随流动性挤兑和风险资产无差别抛售(美债被错杀)。当基于非结构化新闻文本提取的 EPU 指数达到极端高位并开始见顶回落时，标志着恐慌情绪和流动性危机开始衰竭，避险资金将真实回流美债，产生极佳的看多抄底买点。反之，极度自满（EPU极低）后突发上升，标志着平静被打破，通胀或紧缩担忧升温，产生看空脉冲。
    数据: usepuindxd (基于新闻文本提取的美国经济政策不确定性指数)
    触发: 
      看多(避险买入): EPU 252日 Z-Score > 2.0 且 当日值 < 5日均值 且 当日值边际回落
      看空(恐慌抬头): EPU 252日 Z-Score < -2.0 且 当日值 > 5日均值 且 当日值边际上升
    输出: +1.0/-1.0 脉冲信号 (非触发日严格为0.0)
    """

    def __init__(self, z_window=252, smooth_window=5, z_threshold=2.0):
        self.name = 'epu_exhaustion_pulse'
        self.z_window = z_window
        self.smooth_window = smooth_window
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始值为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 获取经济政策不确定性指数并前向填充
        epu = data['usepuindxd'].ffill()
        
        # 计算近期均值用于衰竭判定
        epu_smooth = epu.rolling(window=self.smooth_window).mean()
        
        # 铁律3: 计算 Z-Score，反映预期的极端偏离程度
        epu_mean = epu.rolling(window=self.z_window).mean()
        epu_std = epu.rolling(window=self.z_window).std()
        
        # 避免除以 0
        epu_std = epu_std.replace(0, np.nan)
        z_score = (epu - epu_mean) / epu_std
        
        # 铁律2: 二阶导数 (极值 + 衰竭/边际变化)
        # 条件1: 恐慌极值回落 (看多美债)
        is_panic_exhaustion = (z_score > self.z_threshold) & \
                              (epu < epu_smooth) & \
                              (epu.diff() < 0)
                             
        # 条件2: 自满极值反转 (看空美债)
        is_complacency_reversal = (z_score < -self.z_threshold) & \
                                  (epu > epu_smooth) & \
                                  (epu.diff() > 0)
                          
        # 赋值狙击手脉冲信号
        signal.loc[is_panic_exhaustion] = 1.0
        signal.loc[is_complacency_reversal] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"EpuExhaustionPulseFactor(z_window={self.z_window}, smooth_window={self.smooth_window}, z_threshold={self.z_threshold})"