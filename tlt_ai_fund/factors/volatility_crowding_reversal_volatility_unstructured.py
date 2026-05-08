import numpy as np
import pandas as pd

class VolatilityCrowdingReversalFactor:
    """波动率极值与拥挤反转 (volatility/unstructured)

    逻辑: 捕捉跨资产波动率（VIX与黄金波动率）的极端狂飙。在流动性冲击期间（如2020年3月），波动率极端飙升往往伴随跨资产无差别抛售（包括美债）。当且仅当 VIX 触及极端高位并确认跨资产波动率同步回落时，标志流动性危机解除及恐慌情绪衰竭，市场恢复基本面定价，此时被错杀的避险资金回流，触发做多美债(TLT)脉冲信号。
    数据: vixcls (VIX指数), gvzcls (黄金波动率指数)
    触发: VIX 252日 Z-Score > 2.5，且 VIX 与 GVZCLS 均跌破3日均值且当日环比下降 (衰竭确认)
    输出: +1.0 (恐慌衰竭引发的流动性修复买入脉冲)，平时 0.0
    """

    def __init__(self):
        self.name = 'volatility_crowding_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需数据是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 前向填充缺失值
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算 VIX 252日滚动 Z-Score (反映极端恐慌水位)
        vix_mean = vix.rolling(window=252, min_periods=126).mean()
        vix_std = vix.rolling(window=252, min_periods=126).std()
        vix_zscore = (vix - vix_mean) / vix_std
        
        # 衰竭条件 (严格遵守二阶导数铁律：绝对禁止接飞刀，必须等动量确认反转)
        # 1. 均值回落: 低于3日移动平均
        # 2. 边际变化: 当日环比为负
        vix_exhaustion = (vix < vix.rolling(window=3).mean()) & (vix.diff() < 0)
        gvz_exhaustion = (gvz < gvz.rolling(window=3).mean()) & (gvz.diff() < 0)
        
        # 触发多头脉冲 (极端恐慌 + 跨资产恐慌同步衰竭)
        long_trigger = (vix_zscore > 2.5) & vix_exhaustion & gvz_exhaustion
        
        # 赋值信号 (狙击手级零值休眠)
        signal[long_trigger] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"