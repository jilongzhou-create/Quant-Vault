import numpy as np
import pandas as pd

class VolatilityEpuReversalPulseFactor:
    """波动率与政策恐慌衰竭脉冲因子 (volatility/unstructured)

    逻辑: 结合跨资产波动率(VIX, GVZCLS)与基于NLP的非结构化经济政策不确定性(USEPUINDXD)。美债作为安全资产，在“恐慌/不确定性极度狂飙”时可能面临无差别抛售(流动性挤兑)，绝不能接飞刀；只有当波动率与政策不确定性双双达到极值，且边际开始回落(二阶导数为负)时，才确认流动性冲击结束、避险与宽松预期主导，此时触发做多脉冲。反之，在极度自满时突发共振飙升，触发做空脉冲。
    数据: vixcls (标普波动率), usepuindxd (经济政策不确定性), gvzcls (黄金波动率)
    触发: (VIX或EPU 126日Z-Score > 2.0) 且 VIX开始衰竭(较昨日下跌且低于3日均值) 且 (EPU或黄金波动率回落) -> +1.0
          (VIX或EPU 126日Z-Score < -1.5) 且 VIX突然发散(较昨日上涨且高于3日均值) 且 (EPU或黄金波动率飙升) -> -1.0
    输出: [-1.0, 1.0] 的狙击型脉冲信号
    """

    def __init__(self, window: int = 126):
        self.name = 'volatility_epu_reversal_pulse'
        self.window = window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'usepuindxd', 'gvzcls']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        # 前向填充处理缺失值，确保对齐
        df = data[required_cols].ffill()
        
        vix = df['vixcls']
        epu = df['usepuindxd']
        gvz = df['gvzcls']
        
        # 计算动量水位极值 (Z-Score)
        vix_mean = vix.rolling(self.window).mean()
        vix_std = vix.rolling(self.window).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)
        
        epu_mean = epu.rolling(self.window).mean()
        epu_std = epu.rolling(self.window).std()
        epu_z = (epu - epu_mean) / (epu_std + 1e-8)
        
        # 二阶导数与边际变化 (Anti-Catch-Falling-Knife)
        # VIX 衰竭条件: 较昨日回落 且 跌破3日均线
        vix_falling = (vix.diff() < 0) & (vix < vix.rolling(3).mean())
        epu_falling = (epu.diff() < 0)
        gvz_falling = (gvz.diff() < 0)
        
        # VIX 飙升条件: 较昨日上涨 且 突破3日均线
        vix_rising = (vix.diff() > 0) & (vix > vix.rolling(3).mean())
        epu_rising = (epu.diff() > 0)
        gvz_rising = (gvz.diff() > 0)
        
        # 多头脉冲：恐慌极值 + 边际衰竭 + 跨资产确认
        panic_extreme = (vix_z > 2.0) | (epu_z > 2.0)
        long_cond = panic_extreme & vix_falling & (epu_falling | gvz_falling)
        
        # 空头脉冲：极度自满 + 突发恐慌惊醒 + 跨资产确认
        complacency_extreme = (vix_z < -1.5) | (epu_z < -1.5)
        short_cond = complacency_extreme & vix_rising & (epu_rising | gvz_rising)
        
        # 信号赋值 (零值休眠，仅满足条件输出脉冲)
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window})"