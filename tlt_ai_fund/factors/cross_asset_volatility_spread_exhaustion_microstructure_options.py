import numpy as np
import pandas as pd

class CrossAssetVolatilitySpreadExhaustionFactor:
    """跨资产隐含波动率价差衰竭反转 (microstructure/options)

    逻辑: 当美股恐慌(VIX)相对于黄金避险波动(GVZ)出现极端背离并飙升至极值时, 意味着发生无差别的跨资产流动性抛售。一旦这种极端的波动率差值开始回落(衰竭), 标志着流动性挤兑见顶, 资金将重新配置并大规模回流美债避险, 构成极强的美债(TLT)看多脉冲信号。
    数据: vixcls (VIX指数), gvzcls (黄金ETF隐含波动率)
    触发: (VIX - GVZCLS)的252日Z-Score > 2.5 AND 当前差值 < 过去3日均值
    输出: +1.0 (极短期看多脉冲), 常态为 0.0
    """

    def __init__(self, window: int = 252, z_threshold: float = 2.5, exhaust_window: int = 3):
        self.name = 'cross_asset_vol_spread_exhaustion'
        self.window = window
        self.z_threshold = z_threshold
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        signal.name = self.name
        
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal

        # 获取期权隐含波动率数据并处理缺失值
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产波动率价差 (股票恐慌溢价)
        vol_spread = vix - gvz
        
        # 计算滚动 Z-Score 以衡量价差的极端程度
        roll_mean = vol_spread.rolling(window=self.window).mean()
        roll_std = vol_spread.rolling(window=self.window).std()
        roll_std = roll_std.replace(0, np.nan)  # 避免除零
        
        z_score = (vol_spread - roll_mean) / roll_std
        
        # 二阶导数条件: 动量衰竭 (价差开始从极值回落)
        exhaustion_condition = vol_spread < vol_spread.rolling(window=self.exhaust_window).mean()
        
        # 严格的脉冲触发条件: 极值 AND 衰竭
        trigger_long = (z_score > self.z_threshold) & exhaustion_condition
        
        # 输出脉冲信号
        signal.loc[trigger_long] = 1.0
        
        return signal

    def __repr__(self):
        return f"CrossAssetVolatilitySpreadExhaustionFactor(window={self.window}, z_threshold={self.z_threshold}, exhaust_window={self.exhaust_window})"