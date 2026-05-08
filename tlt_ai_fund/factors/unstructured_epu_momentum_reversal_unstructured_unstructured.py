import numpy as np
import pandas as pd

class UnstructuredEpuMomentumReversalFactor:
    """新闻政策不确定性动量反转因子 (unstructured/unstructured)

    逻辑: 衡量基于新闻提炼的经济政策不确定性(EPU)的短期加速。当不确定性在短期内极度飙升(产生避险买盘)并达到极端(Z>2.5)且开始见顶回落时, 避险情绪退潮, 长端美债失去支撑, 生成看空脉冲(-1.0); 反之, 不确定性降至极度自满冰点且开始反弹时, 新闻恐慌突增, 避险资金重新追捧长端美债, 生成看多脉冲(+1.0)。
    数据: usepuindxd (Daily News Economic Policy Uncertainty Index)
    触发: 5日动量的 252日 Z-Score 突破 ±2.5 且开始向反方向拐头 (极值 + 二阶导数衰竭)
    输出: 脉冲信号 [-1.0, 1.0], 常态 0.0
    """

    def __init__(self, diff_window=5, z_window=252, z_threshold=2.5):
        self.name = 'unstructured_epu_momentum_reversal'
        self.diff_window = diff_window
        self.z_window = z_window
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        
        # 铁律3: 边际变化 - 绝对禁止使用绝对水位, 必须用差分捕捉情绪预期突变的瞬间
        # 这里使用5日差分代表一周内政策不确定性情绪的急剧变化(动量)
        epu_diff = epu.diff(self.diff_window)
        
        # 计算动量的长期极值 (252个交易日约一年)
        epu_diff_mean = epu_diff.rolling(window=self.z_window, min_periods=self.z_window//2).mean()
        epu_diff_std = epu_diff.rolling(window=self.z_window, min_periods=self.z_window//2).std()
        
        epu_diff_z = (epu_diff - epu_diff_mean) / epu_diff_std.replace(0, np.nan)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 条件1: Z-score 极端高位 (避险情绪由于坏新闻爆发极度狂热)
        # 条件2: 动量指标开始向下拐头 (坏新闻效应被 Price-in 并开始衰竭)
        short_condition = (epu_diff_z > self.z_threshold) & (epu_diff_z < epu_diff_z.shift(1))
        
        # 条件1: Z-score 极端低位 (新闻极其平淡，市场极度自满)
        # 条件2: 动量指标开始向上拐头 (黑天鹅预警闪烁，避险需求萌芽)
        long_condition = (epu_diff_z < -self.z_threshold) & (epu_diff_z > epu_diff_z.shift(1))
        
        # 铁律1: 零值休眠 (Sniper Pulse) - 常态下为 0.0
        signal.loc[short_condition] = -1.0
        signal.loc[long_condition] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(diff_window={self.diff_window}, z_window={self.z_window}, z_threshold={self.z_threshold})"