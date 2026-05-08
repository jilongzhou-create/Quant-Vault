import numpy as np
import pandas as pd

class CrossAssetVolPanicExhaustionFactor:
    """跨资产波动率恐慌瓦解反转因子 (Volatility/Unstructured)

    逻辑: 结合股市波动率(VIX)与避险黄金波动率(GVZ)监控全面流动性冲击。当双波动率同步极端飙升时，代表无差别抛售(美债亦可能被错杀)；必须等双重波动率从极值(Z-score>2.5)同步开启衰竭回落时，才确认恐慌瓦解、纯正避险资金回流美债，触发安全的抄底买点。
    数据: vixcls (市场隐含波动率), gvzcls (黄金隐含波动率)
    触发: 极值(VIX Z-Score > 2.5 且 GVZ Z-Score > 2.0) + 衰竭(双指标 diff() < 0 且低于3日均值) + 边缘突变(前日不满足今日刚满足)。
    输出: +1.0 为恐慌退潮美债买入脉冲；-1.0 为过度安逸突变(加息/抛售起点)做空脉冲。常态严格 0.0。
    """

    def __init__(self, z_window=252, ma_window=3):
        self.name = 'cross_asset_vol_panic_exhaustion'
        self.z_window = z_window
        self.ma_window = ma_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 数据缺失保护
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 1. 动态极值水位: 计算基于经济交易年(252日)的滚动 Z-Score，捕捉边际极值
        vix_z = (vix - vix.rolling(self.z_window).mean()) / vix.rolling(self.z_window).std()
        gvz_z = (gvz - gvz.rolling(self.z_window).mean()) / gvz.rolling(self.z_window).std()
        
        # 2. 二阶导数衰竭验证 (绝对禁止极值接飞刀)
        # 多头衰竭：双指标动量转负，且脱离极点绝对高峰(低于短均线)
        bull_exhaustion = (
            (vix.diff() < 0) & 
            (gvz.diff() < 0) & 
            (vix < vix.rolling(self.ma_window).mean()) & 
            (gvz < gvz.rolling(self.ma_window).mean())
        )
        
        # 空头惊跳：双指标从过度安逸极低位突然跳升，二阶转正
        bear_shock = (
            (vix.diff() > 0) & 
            (gvz.diff() > 0) & 
            (vix > vix.rolling(self.ma_window).mean()) & 
            (gvz > gvz.rolling(self.ma_window).mean())
        )
        
        # 3. 极值状态判定 (依据正态分布，VIX门槛设2.5极度恐慌，GVZ设2.0作共振确认)
        extreme_panic = (vix_z > 2.5) & (gvz_z > 2.0)
        extreme_complacency = (vix_z < -1.5) & (gvz_z < -1.5)
        
        # 4. 零值休眠与狙击手脉冲过滤 (仅在状态刚逆转的瞬态触发)
        bull_trigger = extreme_panic & bull_exhaustion & ~bull_exhaustion.shift(1).fillna(False)
        bear_trigger = extreme_complacency & bear_shock & ~bear_shock.shift(1).fillna(False)
        
        signal[bull_trigger] = 1.0
        signal[bear_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_window={self.z_window}, ma_window={self.ma_window})"