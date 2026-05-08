import numpy as np
import pandas as pd

class CrossAssetPanicExhaustionFactor:
    """跨资产波动率恐慌衰竭因子 (microstructure/options)

    逻辑: 捕捉股票市场(VIX)与黄金(GVZ)期权隐含波动率的差值极值。当流动性危机导致股市恐慌远超黄金避险波动时达到极值，并在开始回落的瞬间触发多头脉冲，标志无差别抛售结束，美债将迎来修复性反弹。相反，当差值极度低迷且开始回升时，标志市场从极度自满中惊醒，往往伴随紧缩冲击，美债承压。常态下输出0.0以保持狙击手休眠。
    数据: vixcls, gvzcls
    触发: 126日 Z-Score > 2.0 且 当日值 < 3日均值 -> +1.0；Z-Score < -2.0 且 当日值 > 3日均值 -> -1.0
    输出: +1.0 表示恐慌衰竭看多美债，-1.0 表示自满破裂看空美债，常态为 0.0
    """

    def __init__(self, window=126, z_threshold=2.0, exhaust_window=3):
        self.name = 'cross_asset_panic_exhaustion_pulse'
        self.window = window
        self.z_threshold = z_threshold
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查是否包含必需的数据列
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 提取并前向填充日频数据缺失值
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产恐慌溢价
        spread = vix - gvz
        
        # 计算 Z-Score 以捕捉极端情绪水位
        roll_mean = spread.rolling(window=self.window).mean()
        roll_std = spread.rolling(window=self.window).std()
        roll_std = roll_std.replace(0.0, np.nan)  # 避免除以零
        
        z_score = (spread - roll_mean) / roll_std
        
        # 计算短期均值以验证边际变化 (二阶导数/衰竭条件)
        exhaust_mean = spread.rolling(window=self.exhaust_window).mean()
        
        # 铁律1 & 2: 极值触发 + 衰竭确认
        # 恐慌衰竭: 溢价极度高位 + 开始回落 -> 流动性冲击结束，做多美债
        long_cond = (z_score > self.z_threshold) & (spread < exhaust_mean)
        
        # 自满破裂: 溢价极度低位 + 开始反弹 -> 风险偏好逆转或紧缩开启，做空美债
        short_cond = (z_score < -self.z_threshold) & (spread > exhaust_mean)
        
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold}, exhaust_window={self.exhaust_window})"