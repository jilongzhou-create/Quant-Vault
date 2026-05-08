import numpy as np
import pandas as pd

class FinancialConditionsVolatilityReversalFactor:
    """金融条件与波动率极值交叉反转因子 (volatility/nonlinear)

    逻辑: 结合高频的股市波动率(VIX)与低频的全国金融状况指数(NFCI)。当系统性总恐慌(两者Z-Score之和)达到极端高位时，若出现高频情绪的衰竭(VIX回落)且低频金融压力停止恶化(NFCI边际改善)，则确认为极度拥挤崩塌，此时救市预期升温，产生脉冲式买入美债(TLT)信号。反向自满瓦解则做空。
    数据: vixcls, nfci
    触发: (VIX_Z + NFCI_Z > 2.5) AND VIX < 3日均值 AND VIX.diff() < 0 AND NFCI.diff(5) <= 0 -> +1.0
    输出: 严格脉冲型 [-1.0, 1.0]，常态下休眠恒为 0.0。
    """

    def __init__(self, window=252, min_periods=126):
        self.name = 'financial_conditions_volatility_reversal'
        self.window = window
        self.min_periods = min_periods
        
        # 极值触发阈值
        self.panic_threshold = 2.5 
        self.complacency_threshold = -2.0

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'nfci']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        # 前向填充缺失值以对齐日频
        vix = data['vixcls'].ffill()
        nfci = data['nfci'].ffill()
        
        # 1. 计算长期 Z-Score (防范绝对水位的影响)
        vix_mean = vix.rolling(window=self.window, min_periods=self.min_periods).mean()
        vix_std = vix.rolling(window=self.window, min_periods=self.min_periods).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std
        
        nfci_mean = nfci.rolling(window=self.window, min_periods=self.min_periods).mean()
        nfci_std = nfci.rolling(window=self.window, min_periods=self.min_periods).std().replace(0, np.nan)
        nfci_z = (nfci - nfci_mean) / nfci_std
        
        # 交叉构建系统性宏观状态指数 (非线性叠加)
        # NFCI > 0 代表紧缩，VIX > 0 代表恐慌，相加代表总体宏观压力
        macro_stress_index = vix_z + nfci_z
        
        # 2. 核心铁律: 二阶导数与高频情绪衰竭 (防接飞刀)
        vix_ma3 = vix.rolling(window=3, min_periods=1).mean()
        vix_diff = vix.diff()
        
        # 多头衰竭：波动率跌破3日均值且当日实质性回落
        vix_exhaustion_long = (vix_diff < 0) & (vix < vix_ma3)
        
        # 空头反转：波动率突破3日均值且当日实质性抬头
        vix_exhaustion_short = (vix_diff > 0) & (vix > vix_ma3)
        
        # 3. 核心铁律: 边际变化确认 (处理阶梯状低频数据的瞬时变化)
        # nfci 是周频数据，diff(5) 提取周度边际动量变化，禁止直接对比绝对值
        nfci_diff5 = nfci.diff(5)
        
        # 4. 综合脉冲触发逻辑
        long_cond = (
            (macro_stress_index > self.panic_threshold) & 
            vix_exhaustion_long & 
            (nfci_diff5 <= 0)
        )
        
        short_cond = (
            (macro_stress_index < self.complacency_threshold) & 
            vix_exhaustion_short & 
            (nfci_diff5 >= 0)
        )
        
        # 狙击手级脉冲赋值，其余时间默认休眠为 0.0
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window})"