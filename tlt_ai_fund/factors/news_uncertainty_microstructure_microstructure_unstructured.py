import numpy as np
import pandas as pd

class NewsUncertaintyMicrostructureFactor:
    """新闻政策不确定性微观脉冲因子 (microstructure/unstructured)

    逻辑: 基于新闻文本的经济政策不确定性指数(USEPUINDXD)属于典型的非结构化衍生微观情绪数据。当政策不确定性短期剧烈飙升伴随流动性恐慌时不能盲目买入美债，必须等不确定性微观动量达到极值(Z-Score > 2.5)且开始均值回归回落(当前值<3日均值)时，确认恐慌抛售衰竭才输出看多脉冲；反之极度乐观后的微观反弹则输出看空脉冲。
    数据: usepuindxd (经济政策不确定性日度指数)
    触发: 3日变化量的252日 Z-Score > 2.5 且 当日值 < 3日均值 → 恐慌衰竭看多脉冲 +1.0；Z-Score < -2.5 且 当日值 > 3日均值 → 乐观终结看空脉冲 -1.0
    输出: 狙击手级别脉冲，符合条件的日输出 +1.0 或 -1.0，其余非触发日严格输出 0.0
    """

    def __init__(self, window=3, z_window=252, z_threshold=2.5):
        self.name = 'news_uncertainty_microstructure_pulse'
        self.window = window
        self.z_window = z_window
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (初始化为全0)
        signal = pd.Series(0.0, index=data.index)
        
        # 字段检查
        if 'usepuindxd' not in data.columns:
            signal.name = self.name
            return signal
            
        epu = data['usepuindxd'].ffill()
        
        # 铁律3: 边际变化 Only (严禁使用不确定性的绝对水位，只追踪其短期动量跳跃)
        epu_diff = epu.diff(self.window)
        
        # 计算微观动量变化的滚动 Z-Score (防前瞻偏差)
        roll_mean = epu_diff.rolling(window=self.z_window, min_periods=self.z_window // 2).mean()
        roll_std = epu_diff.rolling(window=self.z_window, min_periods=self.z_window // 2).std()
        z_score = (epu_diff - roll_mean) / (roll_std + 1e-8)
        
        # 铁律2: 二阶导数 (必须等待情绪出现边际衰竭)
        epu_roll_mean = epu.rolling(window=self.window).mean()
        exhaustion_long = epu < epu_roll_mean   # 恐慌开始见顶回落
        exhaustion_short = epu > epu_roll_mean  # 乐观开始反转上升
        
        # 组合脉冲触发条件
        long_cond = (z_score > self.z_threshold) & exhaustion_long
        short_cond = (z_score < -self.z_threshold) & exhaustion_short
        
        # 输出脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_window={self.z_window}, z_threshold={self.z_threshold})"