import numpy as np
import pandas as pd

class CrossAssetVolatilityExhaustionFactor:
    """跨资产波动率极值衰竭因子 (microstructure/options)

    逻辑: VIX(美股期权隐含波动率)与GVZ(黄金ETF期权隐含波动率)的价差代表了跨资产流动性恐慌的极致程度。
          当股市恐慌远超黄金避险恐慌时，意味着市场发生无差别抛售流动性危机（通常连美债一起抛）。
          根据二阶导数铁律，当这种跨资产恐慌价差达到极端高位（Z-Score>2.5）并开始衰竭（回落至3日均值下方）时，
          表明无差别抛售结束，避险资金重新回流美债，触发买入脉冲。反之亦然。完全摒弃了连续的高位做多接飞刀逻辑。
    数据: vixcls, gvzcls
    触发: VIX-GVZ Spread 的 252日 Z-Score > 2.5 且 Spread < 3日均值 -> +1.0
          VIX-GVZ Spread 的 252日 Z-Score < -2.5 且 Spread > 3日均值 -> -1.0
    输出: 脉冲信号，+1.0表示流动性抛售结束看多美债，-1.0表示极度自满结束看空美债
    """

    def __init__(self, z_window=252, z_threshold=2.5, exhaust_window=3):
        self.name = 'cross_asset_vol_exhaustion_options'
        self.z_window = z_window
        self.z_threshold = z_threshold
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 必须处理缺失的字段
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            signal.name = self.name
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算期权隐含波动率的跨资产恐慌价差
        spread = vix - gvz
        
        # 铁律3: 计算边际变化与极值水位 (252个交易日大约为一年期的宏观基准)
        roll_mean = spread.rolling(window=self.z_window, min_periods=self.z_window // 2).mean()
        roll_std = spread.rolling(window=self.z_window, min_periods=self.z_window // 2).std()
        
        # 避免除以0
        z_score = (spread - roll_mean) / (roll_std + 1e-6)
        
        # 铁律2: 二阶导数衰竭条件 - 计算短期动量变化
        exhaust_ma = spread.rolling(window=self.exhaust_window, min_periods=2).mean()
        
        # 核心逻辑: 极值出现 AND 恐慌/自满情绪开始反转衰竭
        bull_cond = (z_score > self.z_threshold) & (spread < exhaust_ma)
        bear_cond = (z_score < -self.z_threshold) & (spread > exhaust_ma)
        
        # 铁律1: 零值休眠，仅在脉冲触发日赋值
        signal[bull_cond] = 1.0
        signal[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_window={self.z_window}, z_threshold={self.z_threshold}, exhaust_window={self.exhaust_window})"