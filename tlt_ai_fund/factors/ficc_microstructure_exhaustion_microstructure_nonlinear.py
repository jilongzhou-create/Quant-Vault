import numpy as np
import pandas as pd

class FiccMicrostructureExhaustionFactor:
    """微观结构流动性恐慌衰竭因子 (microstructure/nonlinear)

    逻辑: 结合宏观波动恐慌(VIX)与底层微观流动性/信贷压力(NFCI)的高维非线性交叉。
          绝对禁止在恐慌飙升主跌浪中接飞刀。正确的抄底买点是在流动性危机最严重的时刻，
          多重危机指标同时达到极端水位(Z-Score 极值)，且在边际上开始同步反转回落(衰竭)的瞬间。
          此时表明抛压枯竭或央行已出手干预，输出典型的极短期狙击手脉冲信号看多美债(TLT)。
    数据: vixcls, nfci
    触发: VIX 252日 Z-Score > 2.5 且 VIX < 3日均值 (极端波动+开始衰竭)
          AND NFCI 252日 Z-Score > 2.0 且近3日内 diff < 0 (流动性极度恶化+边际改善)。
    输出: +1.0 (同步衰竭瞬间的做多脉冲), 非常态下严格零值休眠返回 0.0。
    """

    def __init__(self):
        self.name = 'ficc_microstructure_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号严格为 0.0 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 缺失关键依赖列时直接返回全 0
        if 'vixcls' not in data.columns or 'nfci' not in data.columns:
            return signal
            
        # 前向填充以处理数据频率对齐
        vix = data['vixcls'].ffill()
        nfci = data['nfci'].ffill()
        
        # 1. 极值条件 (Anti-Catch-Falling-Knife 条件1)
        # 252个交易日为一年，63天为最小预热窗口
        vix_mean = vix.rolling(window=252, min_periods=63).mean()
        vix_std = vix.rolling(window=252, min_periods=63).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)
        
        nfci_mean = nfci.rolling(window=252, min_periods=63).mean()
        nfci_std = nfci.rolling(window=252, min_periods=63).std()
        nfci_z = (nfci - nfci_mean) / (nfci_std + 1e-8)
        
        # 2. 衰竭与边际变化条件 (二阶导数铁律 + 边际变化铁律)
        # VIX 跌破短期3日均线，代表抛售动能见顶，恐慌情绪出现衰竭
        vix_exhaustion = vix < vix.rolling(window=3).mean()
        
        # NFCI 为低频周度数据阶梯特征，禁止比较绝对水位，必须用 .diff() 捕捉边际改善瞬间
        # 结合 VIX 和 NFCI 见顶可能有短微时间差，使用 rolling(3).min() < 0 
        # 确保在NFCI数据发布确认边际缓解的随后1-3天内允许触发交叉脉冲
        nfci_diff = nfci.diff()
        nfci_exhaustion = nfci_diff.rolling(window=3).min() < 0
        
        # 3. 非线性交叉触发逻辑 (核心脉冲触发点)
        trigger_cond = (vix_z > 2.5) & vix_exhaustion & (nfci_z > 2.0) & nfci_exhaustion
        
        # 当且仅当极端高位并且同步产生边际回落时，输出 +1.0 看多美债脉冲
        signal.loc[trigger_cond] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"