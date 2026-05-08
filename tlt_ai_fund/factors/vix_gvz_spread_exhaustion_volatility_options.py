import numpy as np
import pandas as pd

class VixGvzSpreadExhaustionFactor:
    """波动率极值与拥挤反转 (volatility/options)

    逻辑: 捕捉权益特异性恐慌溢价（VIX与黄金波动率差值）的极端反转。当VIX相比避险黄金的波动率出现极端飙升且开始瓦解时，表明流动性危机未爆发且无差别抛售结束，避险资金加速回流长端美债(TLT，看多)；当该溢价处于极端低谷且开始反弹时，风险资产重燃波动且吸血美债(看空)。因子完全基于短期分布及动量衰竭脉冲触发。
    数据: vixcls, gvzcls
    触发: 63日 Z-Score > 2.5 且 diff < 0 (做多脉冲)；Z-Score < -2.0 且 diff > 0 (做空脉冲)
    输出: 狙击手级别的 [-1.0, 1.0] 脉冲信号，常态严格为 0.0
    """

    def __init__(self):
        self.name = 'vix_gvz_spread_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'gvzcls']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产恐慌溢价: VIX 与 黄金波动率 (GVZ) 的差值
        spread = vix - gvz
        
        # 采用 63 个交易日(约一季度)短期基准计算 Z-Score，反映近期拥挤度水位
        window = 63
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        spread_z = (spread - spread_mean) / (spread_std + 1e-8)
        
        # 二阶导数条件/边际变化：计算单日动量以捕捉衰竭或反转瞬间
        spread_diff = spread.diff()
        vix_diff = vix.diff()
        
        # 多头触发 (Pulse +1.0)
        # 条件1: 恐慌溢价处于极端高位 (短期 Z-Score > 2.5) 
        # 条件2: 开始瓦解 (差值回落 且 VIX 本身同步回落) -> 防止波动率继续飙升接飞刀
        cond_long_extreme = spread_z > 2.5
        cond_long_exhaustion = (spread_diff < 0) & (vix_diff < 0)
        
        # 空头触发 (Pulse -1.0)
        # 条件1: 恐慌溢价极度压缩至历史低点 (短期 Z-Score < -2.0)
        # 条件2: 边际恶化反弹 (差值开始掉头向上 且 VIX 自身开始反弹) 
        cond_short_extreme = spread_z < -2.0
        cond_short_reversal = (spread_diff > 0) & (vix_diff > 0)
        
        # 只在触发条件同时满足的瞬间发出狙击级别的脉冲信号
        signal[cond_long_extreme & cond_long_exhaustion] = 1.0
        signal[cond_short_extreme & cond_short_reversal] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window=63)"