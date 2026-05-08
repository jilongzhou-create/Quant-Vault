import numpy as np
import pandas as pd

class CrossAssetVolReversalFactor:
    """跨资产波动率极值与拥挤反转因子 (volatility/nonlinear)

    逻辑: 监控股市波动率(VIX)与避险资产黄金波动率(GVZCLS)的极端飙升与同步瓦解。当两者同时处于极值并边际回落时，标志全市场非理性恐慌的极度拥挤开始解体、流动性恢复，此时脉冲做多美债(TLT)。当波动率极低且边际飙升时，预示狂热破灭和通胀/加息预期升温，做空美债。
    数据: vixcls (VIX指数), gvzcls (黄金波动率指数)
    触发: 252日 Z-Score > 2.0 且 跨资产指标同时跌破3日均线并下行 (二阶衰竭)
    输出: +1.0 (恐慌同步衰竭, 脉冲看多美债), -1.0 (自满同步破灭, 脉冲看空美债), 常态为 0.0
    """

    def __init__(self, window: int = 252):
        self.name = 'cross_asset_vol_reversal'
        self.window = window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 严格遵守铁律1: 初始信号全为 0.0，零值休眠
        signal = pd.Series(0.0, index=data.index)
        
        # 数据依赖检查
        required_cols = ['vixcls', 'gvzcls']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 获取数据并向下填充处理缺失
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算动量维度的 252日滚动 Z-Score (避免魔法绝对值硬阈值)
        vix_mean = vix.rolling(window=self.window, min_periods=63).mean()
        vix_std = vix.rolling(window=self.window, min_periods=63).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)
        
        gvz_mean = gvz.rolling(window=self.window, min_periods=63).mean()
        gvz_std = gvz.rolling(window=self.window, min_periods=63).std()
        gvz_z = (gvz - gvz_mean) / (gvz_std + 1e-8)
        
        # 严格遵守铁律2与铁律3: 二阶导数衰竭与边际变化确认
        vix_diff = vix.diff()
        gvz_diff = gvz.diff()
        vix_ma3 = vix.rolling(3).mean()
        gvz_ma3 = gvz.rolling(3).mean()
        
        # 多头脉冲：极度恐慌且跨资产同步衰竭 (防止单一资产假衰竭接飞刀)
        # 条件: 波动率Z-Score极高 + 跌破短均线 + 且当日正在下行
        vix_exhausted = (vix < vix_ma3) & (vix_diff < 0)
        gvz_exhausted = (gvz < gvz_ma3) & (gvz_diff < 0)
        long_cond = (vix_z > 2.0) & (gvz_z > 1.5) & vix_exhausted & gvz_exhausted
        
        # 空头脉冲：极度自满且跨资产同步发散飙升 (狂热破灭)
        # 条件: 波动率Z-Score极低 + 突破短均线 + 且当日正在上行
        vix_surging = (vix > vix_ma3) & (vix_diff > 0)
        gvz_surging = (gvz > gvz_ma3) & (gvz_diff > 0)
        short_cond = (vix_z < -1.5) & (gvz_z < -1.5) & vix_surging & gvz_surging
        
        # 赋值狙击级脉冲信号 (+1.0 / -1.0)
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window})"