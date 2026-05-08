import numpy as np
import pandas as pd

class MacroUncertaintyReversalFactor:
    """宏观不确定性极值反转因子 (volatility / unstructured)

    逻辑: 将基于非结构化新闻文本的经济政策不确定性(USEPUINDXD)与市场波动率(VIX)结合。当不确定性或波动率狂飙至极端高位并开始同步回落(衰竭)时，表明恐慌见顶、避险拥挤极度瓦解，触发看多美债的脉冲；反之极度死水平静期被向上打破时，触发看空脉冲。
    数据: usepuindxd, vixcls
    触发: USEPUINDXD 或 VIX 的 252日 Z-Score > 2.5，且两者均小于其 3日均值 (衰竭回落确认)
    输出: +1.0 看多美债脉冲, -1.0 看空美债脉冲, 其余常态时间保持 0.0
    """

    def __init__(self):
        self.name = 'macro_uncertainty_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值休眠信号 (铁律1: 常态下必须为 0.0)
        signal = pd.Series(0.0, index=data.index)
        
        # 字段缺失安全处理
        if 'usepuindxd' not in data.columns or 'vixcls' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        
        # 计算 252日(约一年) 滚动均值和标准差，建立历史水位参考
        epu_mean = epu.rolling(window=252, min_periods=126).mean()
        epu_std = epu.rolling(window=252, min_periods=126).std().replace(0, np.nan)
        epu_zscore = (epu - epu_mean) / epu_std
        
        vix_mean = vix.rolling(window=252, min_periods=126).mean()
        vix_std = vix.rolling(window=252, min_periods=126).std().replace(0, np.nan)
        vix_zscore = (vix - vix_mean) / vix_std

        # 计算 3日均值，用于边际变化和二阶导数衰竭/突破确认 (铁律2 & 铁律3)
        epu_ma3 = epu.rolling(window=3).mean()
        vix_ma3 = vix.rolling(window=3).mean()
        
        # 触发看多脉冲 (+1.0): 恐慌极值 + 跨域同步开始衰竭 (绝对禁止单纯 VIX>40 买入)
        # 条件1: 任意一个指标处于极端恐慌状态 (Z-Score > 2.5)
        # 条件2: 两个指标都出现动量衰竭迹象 (当期值跌破 3日均值)
        long_cond = ((epu_zscore > 2.5) | (vix_zscore > 2.5)) & (epu < epu_ma3) & (vix < vix_ma3)
        
        # 触发看空脉冲 (-1.0): 极度平静被打破 + 边际向上突变
        # 条件1: 任意一个指标处于极度低位死水期 (Z-Score < -2.0)
        # 条件2: 两个指标同步向上突破并开始狂飙 (当期值升破 3日均值)
        short_cond = ((epu_zscore < -2.0) | (vix_zscore < -2.0)) & (epu > epu_ma3) & (vix > vix_ma3)
        
        # 狙击手脉冲赋值
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        # 清理可能产生的 NaN，确保仅输出实数信号
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"