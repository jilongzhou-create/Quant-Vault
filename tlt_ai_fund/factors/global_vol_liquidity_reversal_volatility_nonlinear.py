import numpy as np
import pandas as pd

class GlobalVolLiquidityReversalFactor:
    """全球波动率流动性反转因子 (volatility/nonlinear)

    逻辑: 极端的跨资产波动率飙升通常伴随宏观流动性冲击(如2020年3月)，导致美债作为避险资产也被无差别抛售。当VIX和黄金波动率(GVZ)同时处于极高水位并同步出现衰竭回落时，标志着流动性恐慌瓦解，抛压衰竭，避险买盘将推动美债报复性反弹。常态下必须为0以避免主跌浪消耗。
    数据: vixcls (VIX波动率), gvzcls (黄金波动率指数)
    触发: VIX 252日 Z-Score > 2.5 且 GVZ Z-Score > 2.0，同时两者均出现二阶衰竭 (当日diff < 0 且 VIX 跌破 3日均值)
    输出: +1.0 强烈看多美债 (脉冲型)
    """

    def __init__(self, window_z=252, window_ma=3):
        self.name = 'global_vol_liquidity_reversal'
        self.window_z = window_z
        self.window_ma = window_ma

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始化全0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖数据是否存在
        required_cols = ['vixcls', 'gvzcls']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 数据前向填充以处理非交易日或对齐缺失
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算滚动的 Z-Score 以衡量水位的极端程度
        vix_mean = vix.rolling(self.window_z).mean()
        vix_std = vix.rolling(self.window_z).std().replace(0, 1e-5)
        vix_z = (vix - vix_mean) / vix_std
        
        gvz_mean = gvz.rolling(self.window_z).mean()
        gvz_std = gvz.rolling(self.window_z).std().replace(0, 1e-5)
        gvz_z = (gvz - gvz_mean) / gvz_std
        
        # 条件1: 跨资产波动率极值确认 (非线性交叉)
        # VIX极度恐慌 + 黄金(终极避险资产)也陷入极度高波，代表全球流动性出现问题
        extreme_condition = (vix_z > 2.5) & (gvz_z > 2.0)
        
        # 铁律2 & 铁律3: 二阶导数与边际变化 (防接飞刀)
        # 绝对禁止在波动率上升期买入，必须等动能逆转
        vix_exhaustion = (vix.diff() < 0) & (vix < vix.rolling(self.window_ma).mean())
        gvz_exhaustion = (gvz.diff() < 0)
        
        # 组合逻辑：必须在同一天满足 极值位 + 同步衰竭回落
        trigger_long = extreme_condition & vix_exhaustion & gvz_exhaustion
        
        # 触发脉冲信号
        signal[trigger_long] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window_z={self.window_z}, window_ma={self.window_ma})"