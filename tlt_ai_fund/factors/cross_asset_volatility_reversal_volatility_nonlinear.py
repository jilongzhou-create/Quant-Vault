import numpy as np
import pandas as pd

class CrossAssetVolatilityReversalFactor:
    """跨资产波动率极值与拥挤反转 (volatility/nonlinear)

    逻辑: 监控股市(VIX)与黄金(GVZCLS)两大跨资产波动率的同步极端狂飙。
          单纯波动率飙升时买入美债极易接飞刀（如2022年股债双杀，两者都在抛售）。
          因此必须等双资产波动率不仅触及高位，而且同步开始衰竭回落时，
          才确认去杠杆和无差别抛售结束、资金开始流向长端避险资产，此时触发看多美债脉冲。
    数据: vixcls, gvzcls
    触发: VIX 252日 Z-Score > 2.0 且 diff < 0 (开始衰竭) AND GVZCLS 252日 Z-Score > 1.5 且 diff < 0 (同步确认)
    输出: 满足条件输出 +1.0 (狙击手级脉冲看多)，其余时间严格为 0.0
    """

    def __init__(self, zscore_window: int = 252, vix_extreme: float = 2.0, gvz_extreme: float = 1.5):
        self.name = 'cross_asset_vol_reversal'
        self.zscore_window = zscore_window
        self.vix_extreme = vix_extreme
        self.gvz_extreme = gvz_extreme

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号为全 0.0，满足铁律1: 零值休眠
        signal = pd.Series(0.0, index=data.index)
        
        # 缺失列检查
        required_cols = ['vixcls', 'gvzcls']
        if not all(col in data.columns for col in required_cols):
            signal.name = self.name
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算 252 日滚动 Z-Score (反映常态水平的偏离度)
        vix_mean = vix.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).mean()
        vix_std = vix.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, 1e-6)
        
        gvz_mean = gvz.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).mean()
        gvz_std = gvz.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).std()
        gvz_z = (gvz - gvz_mean) / gvz_std.replace(0, 1e-6)
        
        # 铁律3: 边际变化 (捕捉变化动量)
        vix_diff = vix.diff()
        gvz_diff = gvz.diff()
        
        # 铁律2: 二阶导数 (极值 + 衰竭)
        # 禁止直接买入，必须等到波动率开始回落
        vix_exhaustion = (vix_z > self.vix_extreme) & (vix_diff < 0)
        gvz_exhaustion = (gvz_z > self.gvz_extreme) & (gvz_diff < 0)
        
        # 非线性交叉确认: 跨资产恐慌同步消散
        trigger_long = vix_exhaustion & gvz_exhaustion
        
        # 触发多头脉冲
        signal[trigger_long] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, vix_extreme={self.vix_extreme}, gvz_extreme={self.gvz_extreme})"