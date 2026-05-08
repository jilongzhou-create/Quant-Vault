import numpy as np
import pandas as pd

class VolatilitySpilloverExhaustionFactor:
    """Volatility Spillover Exhaustion (options)

    逻辑: 捕捉股票(VIX)与黄金(GVZ)隐含波动率价差的极端背离及衰竭。当差值极端走高代表流动性恐慌挤兑(抛售一切含美债)，差值见顶回落时危机解除，资金回流避险资产做多美债；当差值极端低迷且开始反弹时，意味着极度自满后风险重估，加息/通胀预期升温，做空美债。
    数据: vixcls, gvzcls
    触发: VIX-GVZ差值的252日Z-Score > 2.5且<3日均值看多；Z-Score < -2.0且>3日均值看空
    输出: +1.0 看多美债(TLT)，-1.0 看空美债(TLT)，其余为 0.0，狙击手级脉冲型
    """

    def __init__(self, window=252, fast_window=3, z_long=2.5, z_short=-2.0):
        self.name = 'vol_spillover_exhaustion'
        self.window = window
        self.fast_window = fast_window
        self.z_long = z_long
        self.z_short = z_short

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 数据存在性检查
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 向前填充处理缺失值
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产波动率价差 (股票隐含风险 vs 避险黄金隐含风险)
        vol_spread = vix - gvz
        
        # 计算滚动 Z-Score (边际极值)
        roll_mean = vol_spread.rolling(window=self.window, min_periods=self.window // 2).mean()
        roll_std = vol_spread.rolling(window=self.window, min_periods=self.window // 2).std()
        
        zscore = (vol_spread - roll_mean) / (roll_std + 1e-6)
        
        # 计算边际变化衰竭 (反飞刀二阶导数条件)
        fast_mean = vol_spread.rolling(window=self.fast_window, min_periods=1).mean()
        
        # 触发条件1: 多头脉冲 (恐慌挤兑达到极值 + 恐慌开始消退/衰竭)
        long_cond = (zscore > self.z_long) & (vol_spread < fast_mean)
        
        # 触发条件2: 空头脉冲 (极度自满/价差极小 + 波动率突然开始飙升/风险重估)
        short_cond = (zscore < self.z_short) & (vol_spread > fast_mean)
        
        # 赋值脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, fast_window={self.fast_window}, z_long={self.z_long}, z_short={self.z_short})"