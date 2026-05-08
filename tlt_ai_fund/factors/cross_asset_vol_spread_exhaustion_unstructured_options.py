import numpy as np
import pandas as pd

class CrossAssetVolSpreadExhaustionFactor:
    """Cross Asset Volatility Spread Exhaustion (unstructured/options)

    逻辑: 通过 VIX (股市期权隐含恐慌) 与 GVZCLS (黄金期权隐含恐慌) 的差值捕捉跨资产恐慌错位。差值极端飙升代表股市发生远超传统避险市场的恐慌，此时通常伴随"卖出一切"的流动性挤兑（美债亦被错杀）。当差值处于极端高位并开始回落时，标志着流动性危机衰竭，传统避险资金重新回流美债，触发胜率极高的做多脉冲。
    数据: vixcls, gvzcls
    触发: VIX-GVZCLS利差的252日Z-Score > 2.5 且该利差开始回落 (< 3日移动平均)
    输出: +1.0 (脉冲型看多美债)
    """

    def __init__(self, window=252, exhaust_window=3, z_threshold=2.5):
        self.name = 'cross_asset_vol_spread_exhaustion'
        self.window = window
        self.exhaust_window = exhaust_window
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 校验必备底层数据
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 边际相对变化: 跨资产隐波利差
        spread = vix - gvz
        
        # 计算统计学极值动态阈值 (Z-Score)
        roll_mean = spread.rolling(window=self.window, min_periods=self.window // 2).mean()
        roll_std = spread.rolling(window=self.window, min_periods=self.window // 2).std()
        
        # 避免极端情况下的除零异常
        roll_std = roll_std.replace(0, np.nan)
        z_score = (spread - roll_mean) / roll_std
        
        # 极值条件: 跨资产错配程度处于罕见高位 (符合铁律2的第一部分)
        extreme_cond = z_score > self.z_threshold
        
        # 衰竭条件: 错配利差开始见顶回落 (符合铁律2的第二部分，防止接飞刀)
        exhaust_mean = spread.rolling(window=self.exhaust_window).mean()
        exhaust_cond = spread < exhaust_mean
        
        # 狙击手脉冲触发 (满足零值休眠铁律)
        trigger_cond = extreme_cond & exhaust_cond
        
        # 只在触发点输出看多脉冲 (+1.0)
        signal.loc[trigger_cond.fillna(False)] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, exhaust={self.exhaust_window}, z_threshold={self.z_threshold})"