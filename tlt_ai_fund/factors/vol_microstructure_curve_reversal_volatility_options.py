import numpy as np
import pandas as pd

class VolMicrostructureCurveReversalFactor:
    """波动率微观结构与曲线反转因子 (volatility/options)

    逻辑: 捕捉跨资产波动率狂飙衰竭与国债收益率曲线动量共振的脉冲机会。当VIX极端飙升后开始回落（跨资产恐慌消退），且短端利率下行导致曲线剧烈变陡（定价宽松）时，触发多头脉冲；当极度自满被打破且曲线平坦化时，触发空头脉冲。
    数据: vixcls, t10y2y, gvzcls
    触发: VIX 252日 Z-Score > 2.5 + VIX与GVZ同步回落(diff<0) + 曲线动量陡峭化(diff(3)>0) -> +1.0
    输出: 脉冲型，[-1.0, 1.0]，非反转日严格休眠为 0.0
    """

    def __init__(self):
        self.name = 'vol_microstructure_curve_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，默认全 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必须的核心字段
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        spread = data['t10y2y'].ffill()
        
        # 1. 计算波动率极值 (252日长周期 Z-Score 识别极端水位)
        vix_mean = vix.rolling(window=252, min_periods=63).mean()
        vix_std = vix.rolling(window=252, min_periods=63).std()
        vix_std = vix_std.replace(0, np.nan) # 防止除零
        vix_z = (vix - vix_mean) / vix_std
        
        # 2. 铁律2: 二阶导数 (极端情况衰竭或反转，绝对禁止极值直接买入)
        vix_exhaustion = vix.diff() < 0
        vix_surge = vix.diff() > 0
        
        # 3. 铁律3: 边际变化 (收益率曲线 3 日平滑动量，捕捉瞬间陡峭化/平坦化)
        spread_mom = spread.diff(3)
        
        # 4. 跨资产确认 (GVZ 黄金波动率作为宏观避险恐慌消退的补充确认)
        gvz_exhaustion = pd.Series(True, index=data.index)
        gvz_surge = pd.Series(True, index=data.index)
        
        if 'gvzcls' in data.columns:
            gvz = data['gvzcls'].ffill()
            gvz_exhaustion = gvz.diff() < 0
            gvz_surge = gvz.diff() > 0
            
        # --- 核心脉冲触发逻辑 ---
        
        # 多头脉冲: VIX极高 (Z > 2.5) + VIX回落 + GVZ回落 (恐慌全面消退) + 曲线开始陡峭化 (短端急速下行，流动性宽松预期)
        bull_trigger = (vix_z > 2.5) & vix_exhaustion & gvz_exhaustion & (spread_mom > 0)
        
        # 空头脉冲: VIX极低 (Z < -1.5, 波动率对数分布的左尾极端自满) + 波动率全面回升 (恐慌初现) + 曲线平坦化 (短端飙升，紧缩恐慌)
        bear_trigger = (vix_z < -1.5) & vix_surge & gvz_surge & (spread_mom < 0)
        
        # 赋值狙击手级脉冲信号
        signal[bull_trigger] = 1.0
        signal[bear_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"