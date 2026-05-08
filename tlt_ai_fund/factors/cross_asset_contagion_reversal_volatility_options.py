import numpy as np
import pandas as pd

class CrossAssetContagionReversalFactor:
    """跨资产传染反转因子 (volatility/options)

    逻辑: 监控极端恐慌传染与极端自满的瓦解。在极端跨资产恐慌(VIX狂飙)期间，债市常因流动性冲击被错杀；当恐慌衰竭(VIX与黄金波动率同步回落)时，避险买盘将推动美债报复性反弹，此时触发脉冲做多。反之，当市场极端自满(VIX极低)被突然打破时，风险平价基金的去杠杆会导致股债双杀，此时触发脉冲做空。
    数据: vixcls (VIX波动率), gvzcls (黄金波动率)
    触发: 
      - 做多: VIX 252日 Z-Score > 2.5 (恐慌极值) AND VIX及GVZ均跌破3日均线 (跨资产恐慌同步衰竭)
      - 做空: VIX 252日 Z-Score < -2.0 (自满极值) AND VIX突破3日均线且当日上涨 (自满惊醒)
    输出: [-1.0, 1.0] 狙击手级脉冲信号
    """

    def __init__(self, window=252, z_long_thresh=2.5, z_short_thresh=-2.0, smooth_window=3):
        self.name = 'cross_asset_contagion_reversal_volatility_options'
        self.window = window
        self.z_long_thresh = z_long_thresh
        self.z_short_thresh = z_short_thresh
        self.smooth_window = smooth_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必要数据是否存在
        required_cols = ['vixcls', 'gvzcls']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 前向填充处理非交易日缺失值 (最多填充5天)
        vix = data['vixcls'].ffill(limit=5)
        gvz = data['gvzcls'].ffill(limit=5)
        
        if vix.isna().all() or gvz.isna().all():
            return signal

        # 计算 VIX 的 252日 Z-Score
        vix_mean = vix.rolling(window=self.window, min_periods=self.window//2).mean()
        vix_std = vix.rolling(window=self.window, min_periods=self.window//2).std()
        
        # 避免除以 0
        vix_std = vix_std.replace(0, np.nan)
        vix_zscore = (vix - vix_mean) / vix_std
        
        # 计算 3日均线作为二阶导数的边际衰竭判定基准
        vix_ma3 = vix.rolling(window=self.smooth_window).mean()
        gvz_ma3 = gvz.rolling(window=self.smooth_window).mean()
        
        # ---------------------------------------------------------------------
        # 多头逻辑 (Anti-Catch-Falling-Knife): 
        # 绝对禁止 VIX > X 直接买入！必须等 VIX 和 黄金波动率 均跌破短期均线，确认跨资产流动性挤兑结束
        # ---------------------------------------------------------------------
        long_condition = (
            (vix_zscore > self.z_long_thresh) &         # 极值: 恐慌处于极端高位
            (vix < vix_ma3) &                           # 衰竭1: 股市恐慌指标掉头向下
            (gvz < gvz_ma3)                             # 衰竭2: 黄金(避险)波动率也同步掉头，确认全面恐慌消退
        )
        
        # ---------------------------------------------------------------------
        # 空头逻辑:
        # 当 VIX 处于极端低位(极端自满)时，风险平价(Risk Parity)基金会大幅加杠杆。
        # 一旦 VIX 突然苏醒飙升，会导致股债同时被程序化抛售(股债双杀)。
        # ---------------------------------------------------------------------
        short_condition = (
            (vix_zscore < self.z_short_thresh) &        # 极值: 波动率处于极端低位(自满)
            (vix > vix_ma3) &                           # 边际突变1: 波动率向上突破均线
            (vix.diff() > 0)                            # 边际突变2: 确认当日动量为正，微观结构发生突变
        )
        
        # 赋值脉冲信号
        signal.loc[long_condition] = 1.0
        signal.loc[short_condition] = -1.0
        
        # 处理可能的 NaN 值
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"CrossAssetContagionReversalFactor(window={self.window}, z_long={self.z_long_thresh}, z_short={self.z_short_thresh})"