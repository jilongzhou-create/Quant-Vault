import numpy as np
import pandas as pd

class EpuMicrostructureExhaustionFactor:
    """经济政策不确定性微观衰竭脉冲因子 (microstructure/unstructured)

    逻辑: 结合基于新闻文本挖掘的经济政策不确定性指数(USEPUINDXD,非结构化数据)与TLT微观成交量(微观结构)。当政策不确定性飙升至极端恐慌水平，且伴随TLT出现微观放量(投降式抛售或抢筹)时，标志恐慌极值。一旦不确定性开始边际回落(衰竭拐点)，标志流动性冲击结束，避险资金将重新定价，触发做多美债脉冲信号。
    数据: usepuindxd (经济政策不确定性指数), volume (TLT成交量)
    触发: 极值条件 (EPU 252日 Z-Score > 2.0 且 Volume 63日 Z-Score > 1.5) + 衰竭条件 (EPU当日值 < 3日均值 且 1日边际变化 < 0)
    输出: +1.0 (恐慌见顶反转看多脉冲), -1.0 (过度乐观极值反转看空脉冲), 其余时间为 0.0 (狙击手模式休眠)
    """

    def __init__(self, epu_z_threshold: float = 2.0, vol_z_threshold: float = 1.5):
        self.name = 'epu_microstructure_exhaustion'
        self.epu_z_threshold = epu_z_threshold
        self.vol_z_threshold = vol_z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需数据字段是否存在
        if 'usepuindxd' not in data.columns or 'volume' not in data.columns:
            return signal
            
        # 数据前向填充以防止缺失
        epu = data['usepuindxd'].ffill()
        vol = data['volume'].ffill()
        
        # 计算 252 日滚动的 EPU Z-Score (反映宏观不确定性极值)
        epu_std = epu.rolling(window=252).std().replace(0, np.nan)
        epu_zscore = (epu - epu.rolling(window=252).mean()) / epu_std
        
        # 计算 63 日(一季度)滚动的 Volume Z-Score (反映微观流动性冲击/投降式成交)
        vol_std = vol.rolling(window=63).std().replace(0, np.nan)
        vol_zscore = (vol - vol.rolling(window=63).mean()) / vol_std
        
        # =======================================================
        # 多头脉冲逻辑: 极端恐慌 + 微观爆量 -> 恐慌边际衰竭买入
        # =======================================================
        
        # 条件1: 指标极端高位 (宏观与微观双极值共振)
        extreme_panic = (epu_zscore > self.epu_z_threshold) & (vol_zscore > self.vol_z_threshold)
        
        # 条件2: 衰竭铁律 (必须回落才触发，禁止高位接飞刀)
        panic_exhaustion = (epu < epu.rolling(window=3).mean()) & (epu.diff() < 0)
        
        # =======================================================
        # 空头脉冲逻辑: 极度乐观 + 微观爆量 -> 不确定性死灰复燃卖出
        # =======================================================
        
        # 条件1: 政策极度确定(贪婪) 且 微观爆量
        extreme_complacency = (epu_zscore < -self.epu_z_threshold) & (vol_zscore > self.vol_z_threshold)
        
        # 条件2: 反转铁律 (不确定性开始抬头，市场可能进入杀估值)
        complacency_reversal = (epu > epu.rolling(window=3).mean()) & (epu.diff() > 0)
        
        # 生成狙击手级别脉冲信号
        signal[extreme_panic & panic_exhaustion] = 1.0
        signal[extreme_complacency & complacency_reversal] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(epu_z_threshold={self.epu_z_threshold}, vol_z_threshold={self.vol_z_threshold})"