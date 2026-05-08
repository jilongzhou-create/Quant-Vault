import numpy as np
import pandas as pd

class ImpliedVolPanicExhaustionFactor:
    """隐含波动率恐慌衰竭因子 (volatility/options)

    逻辑: 监控美股(VIX)与黄金(GVZCLS)隐含衍生品波动率的跨资产挤兑。当任意资产波动率飙升至极端高位(Z-Score>2.5代表流动性危机/极度恐慌)时，美债可能遭到无差别抛售。仅当两者同步跌破3日均线(二阶导数衰竭)时，确认流动性冲击结束，发出脉冲做多信号(避险资金回流)；当跨资产波动率极度拥挤且同时向上突破均线时，发出做空脉冲。
    数据: vixcls, gvzcls
    触发: 极高位(Z-Score > 2.5) + 同步跌破3日均线 -> 脉冲做多(+1.0); 极低位(Z-Score < -1.5) + 同步升破3日均线 -> 脉冲做空(-1.0)
    输出: 严格脉冲型信号, [-1.0, 1.0], 非触发日强制休眠为 0.0
    """

    def __init__(self, window=252, smooth=3, z_high=2.5, z_low=-1.5):
        self.name = 'implied_vol_panic_exhaustion'
        self.window = window
        self.smooth = smooth
        self.z_high = z_high
        self.z_low = z_low

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始值为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需字段
        required_cols = ['vixcls', 'gvzcls']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 提取并填充数据
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算 252 日长窗口 Z-Score
        vix_mean = vix.rolling(self.window).mean()
        vix_std = vix.rolling(self.window).std()
        # 避免除以0
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        gvz_mean = gvz.rolling(self.window).mean()
        gvz_std = gvz.rolling(self.window).std()
        gvz_z = (gvz - gvz_mean) / gvz_std.replace(0, np.nan)
        
        # 铁律2: 二阶导数 - 计算边际动能状态
        vix_smooth = vix.rolling(self.smooth).mean()
        gvz_smooth = gvz.rolling(self.smooth).mean()
        
        # 衰竭确认: 开始回落
        vix_falling = vix < vix_smooth
        gvz_falling = gvz < gvz_smooth
        
        # 平静瓦解: 开始反弹
        vix_rising = vix > vix_smooth
        gvz_rising = gvz > gvz_smooth
        
        # --- 多头触发逻辑 (Panic Exhaustion) ---
        # 条件1: 跨资产中至少有一个陷入极端恐慌
        extreme_panic = (vix_z > self.z_high) | (gvz_z > self.z_high)
        # 条件2: 跨资产恐慌情绪同步进入边际衰竭
        panic_exhaustion = extreme_panic & vix_falling & gvz_falling
        
        # --- 空头触发逻辑 (Complacency Breakout) ---
        # 条件1: 跨资产同时处于极度拥挤的低波动状态 (期权卖方拥挤)
        extreme_complacency = (vix_z < self.z_low) & (gvz_z < self.z_low)
        # 条件2: 平静被意外打破，波动率同步抬头
        complacency_breakout = extreme_complacency & vix_rising & gvz_rising
        
        # 生成狙击手脉冲信号
        signal[panic_exhaustion] = 1.0
        signal[complacency_breakout] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"ImpliedVolPanicExhaustionFactor(window={self.window}, smooth={self.smooth}, z_high={self.z_high}, z_low={self.z_low})"