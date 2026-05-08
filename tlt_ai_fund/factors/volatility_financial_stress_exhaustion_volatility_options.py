import numpy as np
import pandas as pd

class VolatilityFinancialStressExhaustionFactor:
    """波动率与金融压力衰竭因子 (volatility/options)

    逻辑: 结合宏观金融压力(FSI)与微观对冲恐慌(VIX)，当两者均达到极值且VIX开始出现二阶回落时，表明流动性危机或恐慌见顶，市场预期央行即将转向宽松救市，此时做多长端美债(TLT)。当市场极度自满(FSI极低,VIX极低)且波动率开始异动抬头时，表明拥挤的多头瓦解，紧缩预期升温，做空美债。
    数据: vixcls, stlfsi4
    触发: VIX 252日 Z-Score > 2.5 且开始回落 (diff < 0) 且 FSI Z-Score > 1.5 -> +1.0；反之，极度自满且VIX抬头 -> -1.0。
    输出: 脉冲型 [-1.0, 1.0]，正值看多美债，负值看空美债。
    """

    def __init__(self):
        self.name = 'vol_fsi_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns or 'stlfsi4' not in data.columns:
            return signal
            
        # 向前填充缺失值（stlfsi4 为周频，需要 ffill 到日频）
        vix = data['vixcls'].ffill()
        fsi = data['stlfsi4'].ffill()
        
        # 计算 252日 Z-Score，反映相对一年期的极值水平
        vix_roll_mean = vix.rolling(252).mean()
        vix_roll_std = vix.rolling(252).std().replace(0, np.nan)
        vix_z = (vix - vix_roll_mean) / vix_roll_std
        
        fsi_roll_mean = fsi.rolling(252).mean()
        fsi_roll_std = fsi.rolling(252).std().replace(0, np.nan)
        fsi_z = (fsi - fsi_roll_mean) / fsi_roll_std
        
        # 计算 VIX 的二阶导数与边际变化
        vix_ma3 = vix.rolling(3).mean()
        vix_diff1 = vix.diff(1)
        vix_diff3 = vix.diff(3)
        
        # ==========================================
        # 多头触发条件 (恐慌见顶衰竭 -> 做多美债)
        # ==========================================
        # 条件1: 波动率与金融压力均处于极端高位
        long_cond_ext = (vix_z > 2.5) & (fsi_z > 1.5)
        # 条件2: 波动率开始衰竭 (反飞刀二阶导数)
        long_cond_exh = (vix_diff1 < 0) & (vix < vix_ma3)
        
        # ==========================================
        # 空头触发条件 (极度自满被打破 -> 做空美债)
        # ==========================================
        # 条件1: 波动率与金融压力极度自满 (宽松且毫无防备)
        short_cond_ext = (vix_z < -2.0) & (fsi_z < -1.5)
        # 条件2: 波动率开始抬头 (边际恶化)
        short_cond_exh = (vix_diff1 > 0) & (vix > vix_ma3) & (vix_diff3 > 0)
        
        # 信号赋值 (严格遵守脉冲约束)
        signal[long_cond_ext & long_cond_exh] = 1.0
        signal[short_cond_ext & short_cond_exh] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"