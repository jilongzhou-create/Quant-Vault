import numpy as np
import pandas as pd

class CrossAssetOptionsPanicFactor:
    """跨资产期权恐慌极值与衰竭反转因子 (microstructure/options)

    逻辑: 结合权益期权(VIX)与黄金期权(GVZ)隐含波动率构建跨资产恐慌指数。期权隐含波动率代表了市场对风险事件定价的保护成本。当跨资产恐慌极度飙升(Z>2.5)并开始衰竭回落时，意味系统性流动性冲击或通胀恐慌已见顶，长端美债迎来极佳的避险抄底脉冲；反之，当波动率极度低迷(Z<-2.0)并开始抬头，意味市场极度自满的平静期被打破，风险溢价重估引发长债抛售。
    数据: vixcls, gvzcls
    触发: 多头 -> (Z-Score > 2.5 且 当前值 < 3日均值); 空头 -> (Z-Score < -2.0 且 当前值 > 3日均值)
    输出: 脉冲型信号，+1.0表示抄底美债(恐慌消退)，-1.0表示做空美债(风险觉醒)，其余时间0.0休眠。
    """

    def __init__(self, window: int = 252, long_z_thresh: float = 2.5, short_z_thresh: float = -2.0, reverse_window: int = 3):
        self.name = 'cross_asset_options_panic'
        self.window = window
        self.long_z_thresh = long_z_thresh
        self.short_z_thresh = short_z_thresh
        self.reverse_window = reverse_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始值为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 必须处理缺失列的情况
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 构建跨资产波动率合成指标 (代表综合系统性风险溢价)
        composite_vol = vix + gvz
        
        # 计算 252日 Z-Score (代表相较于一年历史的边际偏离极值)
        roll_mean = composite_vol.rolling(window=self.window).mean()
        roll_std = composite_vol.rolling(window=self.window).std()
        
        # 避免除以0导致的无穷大
        roll_std = roll_std.replace(0, np.nan)
        vol_zscore = (composite_vol - roll_mean) / roll_std
        
        # 计算超短期移动均值，作为二阶导数判定的基准线
        vol_short_mean = composite_vol.rolling(window=self.reverse_window).mean()
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 多头脉冲：恐慌极度高涨 (极值) + 恐慌开始回落 (衰竭)
        long_cond = (vol_zscore > self.long_z_thresh) & (composite_vol < vol_short_mean)
        
        # 空头脉冲：过度自满 (极值) + 风险开始抬头 (反转)
        short_cond = (vol_zscore < self.short_z_thresh) & (composite_vol > vol_short_mean)
        
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, long_z={self.long_z_thresh}, short_z={self.short_z_thresh})"