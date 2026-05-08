import numpy as np
import pandas as pd

class VolatilityPivotShockFactor:
    """Volatility Pivot Shock (unstructured/options)

    逻辑: 捕捉期权隐含波动率(VIX)微观结构的极端冲击拐点。美债是终极避险蓄水池，当恐慌(VIX剧烈飙升)见顶并开始衰竭时，流动性抛售危机解除，伴随着央行宽松预期，避险资金大量重新涌入国债，产生做多TLT脉冲；当极度自满(VIX持续剧烈压缩)走到尽头并开始反弹时，避险需求彻底消失且可能伴随经济过热/通胀预期抬头，资金抛售国债追逐高风险资产，产生做空TLT脉冲。
    数据: vixcls
    触发: VIX的5日变化量之252日 Z-Score > 2.5 且绝对值跌破3日均值(恐慌抛售脉冲衰竭 -> +1.0)；Z-Score < -2.5 且绝对值突破3日均值(极度自满脉冲破裂 -> -1.0)
    输出: 狙击手级别的脉冲信号, 输出 +1.0 或 -1.0
    """

    def __init__(self):
        self.name = 'vix_pivot_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号设为全 0.0，严格遵守常态休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns:
            return signal
            
        # 前向填充缺失值
        vix = data['vixcls'].ffill()
        
        # 铁律3: 边际变化 (捕捉5个交易日内预期/情绪的跳跃程度，避免绝对水位导致的连续触发)
        vix_diff5 = vix.diff(5)
        
        # 使用 252 交易日(约1年)计算边际变化的 Z-Score
        roll_mean = vix_diff5.rolling(window=252, min_periods=60).mean()
        roll_std = vix_diff5.rolling(window=252, min_periods=60).std()
        
        zscore = (vix_diff5 - roll_mean) / (roll_std + 1e-8)
        
        # 铁律2: 二阶导数 (必须等待极值动能产生拐点衰竭，绝不逆势接飞刀)
        vix_ma3 = vix.rolling(window=3).mean()
        vix_diff1 = vix.diff(1)
        
        # 恐慌冲击见顶衰竭 -> 资金回补国债 -> 做多 TLT (+1.0)
        # 条件: 短期内波动率极度飙升 + 今天波动率停止飙升且开始明显回落
        long_condition = (
            (zscore > 2.5) & 
            (vix < vix_ma3) & 
            (vix_diff1 < 0)
        )
        
        # 自满情绪极致破裂 -> 抛售无风险资产 -> 做空 TLT (-1.0)
        # 条件: 短期内波动率极度压缩下探 + 今天波动率停止压缩且开始掉头反弹
        short_condition = (
            (zscore < -2.5) & 
            (vix > vix_ma3) & 
            (vix_diff1 > 0)
        )
        
        # 铁律1: 零值休眠，仅在极端拐点瞬间触发脉冲
        signal.loc[long_condition] = 1.0
        signal.loc[short_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"