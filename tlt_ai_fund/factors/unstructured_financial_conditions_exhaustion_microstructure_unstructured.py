import numpy as np
import pandas as pd

class UnstructuredFinancialConditionsExhaustionFactor:
    """金融状况极值衰竭脉冲 (microstructure/unstructured)

    逻辑: 芝加哥联储 NFCI 指数汇总了105项底层微观金融结构数据, 真实反映全市场的流动性挤兑与恐慌压力。对于此类低频阶梯状数据, 绝对禁止根据绝对水位买入(会导致在流动性危机主跌浪中接飞刀)。必须采用二阶导数逻辑: 当流动性高压达到极端极值(Z-Score>2.5), 且在最新周频公布日中首次出现边际回落(diff<0)时, 标志着危机已达极点或美联储干预生效, 此时输出单日做多美债的脉冲信号。
    数据: nfci
    触发: 更新前一日的 252日 Z-Score > 2.5 (处于极度恐慌高位), 且当日边际变动 nfci.diff() < 0 (恐慌见顶衰竭)
    输出: +1.0 (流动性危机解除瞬间, 抄底看多美债)
    """

    def __init__(self):
        self.name = 'unstructured_financial_conditions_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'nfci' not in data.columns:
            return signal
            
        nfci = data['nfci'].ffill()
        
        # 避免全空数据的异常报错
        if nfci.isna().all():
            return signal
            
        # 1. 铁律3: 边际变化。捕捉低频阶梯状数据的更新瞬间
        # 因 NFCI 为周频且日频前填，其 diff() 仅在数据更新的当天非零
        nfci_diff = nfci.diff()
        
        # 2. 计算极值状态 (252交易日窗口的滚动 Z-Score)
        roll_mean = nfci.rolling(window=252, min_periods=21).mean()
        roll_std = nfci.rolling(window=252, min_periods=21).std()
        nfci_zscore = (nfci - roll_mean) / (roll_std + 1e-6)
        
        # 3. 铁律2: 二阶导数防接飞刀 (极高压极值 + 边际衰竭)
        # 条件1: 数据更新的前一日，压力水平仍然处于极端极值状态 (恐慌极点)
        cond_extreme = nfci_zscore.shift(1) > 2.5
        
        # 条件2: 当日发布的最新值出现边际回落，标志着流动性危机实质性见顶缓解
        # 注: 在日频序列中, 前填机制使得仅在更新当日 diff < 0 为 True, 其他前填日均为 False
        # 这从数学结构上天然保证了该信号是一个极度干脆的"单日狙击脉冲"
        cond_exhaustion = nfci_diff < 0.0
        
        # 只有在极端恐慌衰竭的拐点瞬间，才触发看多脉冲
        trigger = cond_extreme & cond_exhaustion
        
        signal[trigger] = 1.0
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"