import numpy as np
import pandas as pd

class CrossAssetVolExhaustionFactor:
    """波动率极值与拥挤反转 (volatility/nonlinear)

    逻辑: 结合VIX与黄金波动率(GVZ)衡量跨资产恐慌。当全面恐慌达到极值且同步回落时，表明流动性危机衰竭，做多美债(反转脉冲)；当极度自满结束且波动率开始飙升时，表明流动性收紧与紧缩风险发酵，做空美债。
    数据: vixcls, gvzcls
    触发: 252日联合Z-Score极值 (>1.5或<-1.2) + 波动率二阶导数反转(diff<0且落于3日均线下方)
    输出: +1.0 (恐慌衰竭脉冲, 多头), -1.0 (自满打破脉冲, 空头), 其余常态时间为 0.0
    """

    def __init__(self):
        self.name = 'cross_asset_vol_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，默认返回 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        if vix.isna().all() or gvz.isna().all():
            return signal
            
        # 计算 252日 Z-Score，反映宏观一年的极值情况
        vix_std = vix.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        gvz_std = gvz.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        
        vix_z = (vix - vix.rolling(window=252, min_periods=60).mean()) / vix_std
        gvz_z = (gvz - gvz.rolling(window=252, min_periods=60).mean()) / gvz_std
        
        # 跨资产波动率压力合成 (VIX 代表股票/信用层面，GVZ 代表实物避险/法币信用层面)
        vol_stress = (vix_z + gvz_z) / 2.0
        
        # 铁律3: 边际变化 - 获取均值以及边际差分变化
        vix_ma3 = vix.rolling(window=3).mean()
        gvz_ma3 = gvz.rolling(window=3).mean()
        
        # 铁律2: 二阶导数 - 衰竭确认 (避免接飞刀)
        vix_exhaustion = (vix < vix_ma3) & (vix.diff() < 0)
        gvz_exhaustion = (gvz < gvz_ma3) & (gvz.diff() < 0)
        
        # 二阶导数 - 突变确认 (自满情绪破裂)
        vix_surge = (vix > vix_ma3) & (vix.diff() > 0)
        gvz_surge = (gvz > gvz_ma3) & (gvz.diff() > 0)
        
        # 触发多头：合成波动率处于极高位 (>1.5 std)，且跨资产恐慌情绪同步出现实质性回落
        long_cond = (vol_stress > 1.5) & vix_exhaustion & gvz_exhaustion
        
        # 触发空头：市场处于极度自满期 (<-1.2 std)，且跨资产波动率同步开始抬升
        short_cond = (vol_stress < -1.2) & vix_surge & gvz_surge
        
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"