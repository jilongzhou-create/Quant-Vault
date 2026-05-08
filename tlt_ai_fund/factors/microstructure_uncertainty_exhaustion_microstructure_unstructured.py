import numpy as np
import pandas as pd

class MicrostructureUncertaintyExhaustionFactor:
    """Microstructure & Uncertainty Panic Exhaustion (microstructure/unstructured)

    逻辑: 政策不确定性(EPU)突升往往伴随TLT微观成交量的应激爆量，反映了市场面对非结构化突发新闻时的恐慌抛售或流动性冲击。当不确定性边际冲击极值(Z-Score>1.25)或成交量极值出现，且随后成交量迅速回落至均值以下、不确定性环比降低(恐慌情绪与微观动量双重衰竭)时，表明左侧抛压出尽，形成高胜率的美债抄底脉冲。
    数据: usepuindxd (经济政策不确定性指数), volume (TLT ETF成交量)
    触发: (EPU 5日差分的 63日 Z-Score > 1.25 或 成交量 Z-Score > 1.25) 且 EPU当日环比回落(diff<0) 且 成交量低于5日均值
    输出: +1.0 (恐慌冲击出尽，看多美债)
    """

    def __init__(self, window: int = 63, z_threshold: float = 1.25, exhaust_window: int = 5):
        self.name = 'microstructure_uncertainty_exhaustion'
        self.window = window              # 季度滚动窗口(约63个交易日)
        self.z_threshold = z_threshold    # 极值触发阈值，设定在约前10%分布以保证5-15%的触发率
        self.exhaust_window = exhaust_window  # 微观动量衰竭窗口(1周/5日)

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (初始常态下严格返回0.0)
        signal = pd.Series(0.0, index=data.index)
        
        # 校验必备字段
        if 'usepuindxd' not in data.columns or 'volume' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        vol = data['volume'].ffill()
        
        # 铁律3: 边际变化 (绝对禁止直接用EPU绝对水位，取5日变化量捕捉跳跃突变)
        epu_diff = epu.diff(self.exhaust_window)
        
        # 计算季度内 EPU 边际冲击的 Z-Score
        epu_diff_mean = epu_diff.rolling(self.window).mean()
        epu_diff_std = epu_diff.rolling(self.window).std()
        epu_z = (epu_diff - epu_diff_mean) / (epu_diff_std + 1e-8)
        
        # 计算微观结构中成交量放大的 Z-Score
        vol_mean = vol.rolling(self.window).mean()
        vol_std = vol.rolling(self.window).std()
        vol_z = (vol - vol_mean) / (vol_std + 1e-8)
        
        # 条件1: 极值判定 (不确定性新闻边际突升 导致 交易拥挤流动性骤变)
        extreme_panic = (epu_z > self.z_threshold) | (vol_z > self.z_threshold)
        
        # 铁律2: 二阶导数 (拒绝接飞刀，必须看到极端状态实质性降温才介入)
        # 衰竭条件A: 不确定性情绪今日开始降温 (环比差分为负)
        epu_exhaustion = epu.diff(1) < 0
        # 衰竭条件B: 微观成交萎缩，抛压停止放量 (低于1周移动平均)
        vol_exhaustion = vol < vol.rolling(self.exhaust_window).mean()
        
        # 脉冲触发: 极值 + 衰竭同时满足
        trigger = extreme_panic & epu_exhaustion & vol_exhaustion
        
        # 脉冲发车 (+1.0，正向Carry资产做多)
        signal.loc[trigger] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold}, exhaust_window={self.exhaust_window})"