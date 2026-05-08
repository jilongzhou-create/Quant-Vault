import numpy as np
import pandas as pd

class FinancialPanicReversalNonlinearFactor:
    """Financial Panic Reversal Factor (volatility/nonlinear)

    逻辑: 资产波动率(VIX)与系统金融状况压力(NFCI)在达到极值时，通常伴随流动性枯竭与"现金为王"，此时强行买入美债容易死于流动性抛售(接飞刀)。只有当恐慌极值与系统压力双双出现边际缓解(二阶导数转负)时，才确认流动性危机解除，避险资金与基本面宽松预期主导，触发美债(TLT)脉冲式做多点。反之极度自满且边际恶化时看空美债。
    数据: vixcls, nfci
    触发: VIX一年期Z-Score > 1.5 且 NFCI Z-Score > 1.0，同时 VIX 3日动量 < 0 且 NFCI 5日(周度)动量 < 0，触发 +1.0。空头反转触发 -1.0。
    输出: 严格脉冲型信号，+1.0(看多TLT), -1.0(看空TLT), 其余状态休眠(0.0)。
    """

    def __init__(self):
        self.name = 'financial_panic_reversal_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需卫星因子数据是否存在
        if 'vixcls' not in data.columns or 'nfci' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        nfci = data['nfci'].ffill()
        
        # 经济学含义: 计算 252个交易日(约1年) 的宏观偏离水位 Z-Score
        vix_mean = vix.rolling(252).mean()
        vix_std = vix.rolling(252).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        nfci_mean = nfci.rolling(252).mean()
        nfci_std = nfci.rolling(252).std()
        nfci_z = (nfci - nfci_mean) / nfci_std.replace(0, np.nan)
        
        # 边际变化与衰竭确认 (严格遵守二阶导数防飞刀铁律)
        # VIX采用3日短期动量，NFCI是偏低频数据采用5日(约一周)动量
        vix_exhaust_long = vix.diff(3) < 0
        nfci_exhaust_long = nfci.diff(5) < 0
        
        vix_exhaust_short = vix.diff(3) > 0
        nfci_exhaust_short = nfci.diff(5) > 0
        
        # 多头触发：资产恐慌与系统流动性枯竭均处于高水位，且双双边际缓解
        long_cond = (
            (vix_z > 1.5) & 
            (nfci_z > 1.0) & 
            vix_exhaust_long & 
            nfci_exhaust_long
        )
        
        # 空头触发：资产极度自满与系统流动性极其泛滥(负偏离)，且双双开始边际恶化收紧
        short_cond = (
            (vix_z < -1.0) & 
            (nfci_z < -1.0) & 
            vix_exhaust_short & 
            nfci_exhaust_short
        )
        
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"