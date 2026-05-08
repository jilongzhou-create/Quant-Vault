import numpy as np
import pandas as pd

class MicrostructurePlumbingStressExhaustionFactor:
    """Microstructure Plumbing Stress Exhaustion (microstructure/nonlinear)

    逻辑: 当宏观恐慌(VIX)与微观金融管道压力(NFCI)同步恶化产生非线性共振极值时，表明发生流动性危机(Dash for Cash)。必须等待恐慌指标见顶且停止恶化(VIX首次跌破3日均值 且 NFCI停止上升)，确认为流动性拐点(美联储等干预生效)，此时极短期脉冲做多美债(TLT)。
    数据: vixcls, nfci
    触发: VIX与NFCI的Z-Score乘积 > 2.0 + VIX当日首次向下穿越3日均值 + NFCI周度动量非正
    输出: +1.0 (流动性恐慌见顶拐点，脉冲看多)
    """

    def __init__(self):
        self.name = 'microstructure_plumbing_stress_exhaustion_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 严格遵守铁律1: 常态下信号必须为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 必须处理数据缺失的情况
        if 'vixcls' not in data.columns or 'nfci' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        nfci = data['nfci'].ffill()
        
        # 1. 计算长期 Z-Score (252日交易日，反映经济学一年周期)
        vix_z = (vix - vix.rolling(window=252).mean()) / vix.rolling(window=252).std()
        nfci_z = (nfci - nfci.rolling(window=252).mean()) / nfci.rolling(window=252).std()
        
        # 2. 挖掘方法C: 非线性特征交叉
        # 仅当两指标均在均值之上(压力恶化区)时计算乘积，放大微观与宏观的共振效应，过滤噪音
        stress_cross = np.where((vix_z > 0) & (nfci_z > 0), vix_z * nfci_z, 0.0)
        stress_cross = pd.Series(stress_cross, index=data.index)
        
        # 3. 极值状态识别 (阈值 2.0 兼顾极值尾部属性与 5%-15% 的 Trigger Rate 目标)
        is_extreme = stress_cross > 2.0
        
        # 赋予极值状态 5 日记忆窗口，以容纳宏观高频数据(VIX)与微观低频数据(NFCI)的时间差
        extreme_memory = is_extreme.rolling(window=5).max() > 0
        
        # 4. 严格遵守铁律2与铁律3: 二阶导数衰竭与边际变化 (Anti-Catch-Falling-Knife)
        
        # 狙击手脉冲条件: 仅在 VIX "首次" 跌破 3日均值的那一天触发，防止连续输出非零值
        vix_mean = vix.rolling(window=3).mean()
        vix_exhaustion = (vix < vix_mean) & (vix.shift(1) >= vix_mean.shift(1))
        
        # 边际变化条件: NFCI为低频阶梯数据，绝对禁止只看绝对水位，考察其周度边际变化 (停止上升)
        nfci_exhaustion = nfci.diff(5) <= 0
        
        # 5. 脉冲触发
        # 只有在近期发生过流动性恐慌共振(极值条件)，且今天双指标均出现衰竭信号时才输出
        buy_trigger = extreme_memory & vix_exhaustion & nfci_exhaustion
        
        # 赋值 +1.0 看多美债
        signal[buy_trigger] = 1.0
        
        signal.name = self.name
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"