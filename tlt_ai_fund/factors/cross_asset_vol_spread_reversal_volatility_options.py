import numpy as np
import pandas as pd

class CrossAssetVolSpreadReversalFactor:
    """跨资产波动率利差反转脉冲因子 (volatility/options)

    逻辑: 股市恐慌(VIX)通常比黄金恐慌(GVZ)飙升更剧烈，当 VIX 或 VIX-GVZ 利差达到极端高位(对冲极度拥挤)且开始回落时，标志着跨资产流动性危机消退，市场焦点转向博弈央行宽松预期，此时为美债(TLT)强劲买入脉冲。反之，极度自满环境被打破时为看空脉冲。
    数据: vixcls, gvzcls
    触发: 252日 Z-Score > 2.5 且开始衰竭回落 (跌破3日均值且当日diff<0) 触发看多；Z-Score < -2.0 且向上突破 触发看空。
    输出: +1.0 看多TLT, -1.0 看空TLT, 常态输出 0.0 零值休眠。
    """

    def __init__(self, window=252, smooth=3, z_long=2.5, z_short=-2.0):
        self.name = 'cross_asset_vol_spread_reversal'
        self.window = window
        self.smooth = smooth
        self.z_long = z_long
        self.z_short = z_short

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 必须验证所需数据字段是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 前向填充缺失值，避免对齐导致的空洞
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产波动率利差
        spread = vix - gvz
        
        # 计算 252 日 Rolling Z-Score
        vix_mean = vix.rolling(window=self.window, min_periods=self.window//2).mean()
        vix_std = vix.rolling(window=self.window, min_periods=self.window//2).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        spread_mean = spread.rolling(window=self.window, min_periods=self.window//2).mean()
        spread_std = spread.rolling(window=self.window, min_periods=self.window//2).std()
        spread_z = (spread - spread_mean) / spread_std.replace(0, np.nan)
        
        # 极值条件判断 (铁律1: 狙击手脉冲)
        high_panic = (vix_z > self.z_long) | (spread_z > self.z_long)
        low_panic = (vix_z < self.z_short) | (spread_z < self.z_short)
        
        # 衰竭条件判断 (铁律2: 二阶导数 + 铁律3: 边际变化)
        vix_smooth = vix.rolling(self.smooth, min_periods=1).mean()
        spread_smooth = spread.rolling(self.smooth, min_periods=1).mean()
        
        # 恐慌衰竭: 波动率绝对值跌破短期均线，且边际差分为负
        panic_exhausted = (
            (vix < vix_smooth) & 
            (spread < spread_smooth) & 
            (vix.diff() < 0) & 
            (spread.diff() < 0)
        )
        
        # 自满衰竭: 极度低波后突然抬头突破均线，且边际差分为正
        complacency_exhausted = (
            (vix > vix_smooth) & 
            (spread > spread_smooth) & 
            (vix.diff() > 0) & 
            (spread.diff() > 0)
        )
        
        # 只有在 极值条件 与 衰竭条件 同时满足时才输出非零脉冲信号
        long_signal = high_panic & panic_exhausted
        short_signal = low_panic & complacency_exhausted
        
        signal[long_signal] = 1.0
        signal[short_signal] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, smooth={self.smooth}, z_long={self.z_long})"