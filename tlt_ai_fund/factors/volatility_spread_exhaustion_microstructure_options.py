import numpy as np
import pandas as pd

class VolatilitySpreadExhaustionFactor:
    """Volatility Spread Exhaustion (microstructure/options)

    逻辑: 捕捉美股波动率(VIX)相对黄金波动率(GVZ)的极端溢价与衰竭。当 VIX-GVZ 差值出现极大偏离后回落，代表权益市场的特异性极度恐慌(或流动性冲击)见顶退潮，此时美债市场的流动性抛售压制解除，长端美债迎来绝佳的反弹抄底契机。使用脉冲非连续信号以避免在波动率主升浪中接飞刀。
    数据: vixcls, gvzcls
    触发: VIX-GVZ 差值的 252 日 Z-Score > 2.0，且差值当日环比下降且低于3日均值。
    输出: +1.0 表示跨资产恐慌溢价开始衰竭，看多美债(TLT)反弹；常态严格为 0.0。
    """

    def __init__(self, window=252, z_threshold=2.0, smooth_window=3):
        self.name = 'vol_spread_exhaustion_factor'
        self.window = window
        self.z_threshold = z_threshold
        self.smooth_window = smooth_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 序列, 严格遵守零值休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 字段校验
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 微观结构特征: 期权隐含波动率差值 (Volatility Spread)
        vol_spread = vix - gvz
        
        # 计算动量差值的 252 日 Z-Score
        spread_mean = vol_spread.rolling(self.window, min_periods=self.window//2).mean()
        # 避免除以零
        spread_std = vol_spread.rolling(self.window, min_periods=self.window//2).std().replace(0, np.nan)
        spread_zscore = (vol_spread - spread_mean) / spread_std
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 条件1: 跨资产恐慌溢价处于极端高位
        condition_extreme = spread_zscore > self.z_threshold
        
        # 条件2: 恐慌开始实质性衰竭 (边际变化为负)
        spread_ma3 = vol_spread.rolling(self.smooth_window).mean()
        condition_exhaustion = (vol_spread < spread_ma3) & (vol_spread.diff() < 0)
        
        # 组合触发: 狙击手级别的脉冲信号
        buy_signal = condition_extreme & condition_exhaustion
        
        signal[buy_signal] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold}, smooth_window={self.smooth_window})"