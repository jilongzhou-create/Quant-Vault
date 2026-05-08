import numpy as np
import pandas as pd

class OptionsVolMomentumExhaustionFactor:
    """Cross-Asset Volatility Momentum Reversal (microstructure/options)

    逻辑: 捕捉跨资产期权隐含波动率动量的极端脉冲与衰竭，属于典型的狙击手级脉冲因子。
          相比于传统的绝对水位因子，动量(边际变化)因子能提供极强的正交收益。
          1. 当VIX短期飙升(5日动量Z-Score>2.5)且开始单日回落时，标志着股市流动性恐慌初步见顶，避险资金将回流美债，输出+1.0。
          2. 当黄金波动率(GVZ)动量极度萎缩(Z-Score<-2.5，代表通胀/尾部风险极度自满)且开始单日反弹时，标志着通胀/尾部风险重新定价，美债将承压，输出-1.0。
    数据: vixcls, gvzcls
    触发: 5日变化率的42日Z-Score极值 + 单日反转衰竭
    输出: 脉冲型信号 [-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'options_vol_momentum_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 前向填充缺失值，防止 NaN 干扰
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝对禁止直接使用波动率水位，必须计算其 5 日动量
        vix_mom = vix.diff(5)
        gvz_mom = gvz.diff(5)
        
        # 计算动量的 42 日 Z-Score 
        # (使用 42 日即约两个月的较短窗口，确保对微观脉冲的敏感度，达到 5%-15% 的健康触发率)
        vix_mom_mean = vix_mom.rolling(42).mean()
        vix_mom_std = vix_mom.rolling(42).std().replace(0, np.nan)
        vix_mom_z = (vix_mom - vix_mom_mean) / vix_mom_std
        
        gvz_mom_mean = gvz_mom.rolling(42).mean()
        gvz_mom_std = gvz_mom.rolling(42).std().replace(0, np.nan)
        gvz_mom_z = (gvz_mom - gvz_mom_mean) / gvz_mom_std
        
        # 铁律2: 二阶导数 (极值 + 衰竭)
        # 条件1：股市恐慌动量处于极端极值 (Z > 2.5) 且当日恐慌开始回落 (diff < 0) -> 做多 TLT
        long_cond = (vix_mom_z > 2.5) & (vix.diff(1) < 0)
        
        # 条件2：黄金/通胀恐慌极度自满 (Z < -2.5) 且当日开始反弹重燃 (diff > 0) -> 做空 TLT
        short_cond = (gvz_mom_z < -2.5) & (gvz.diff(1) > 0)
        
        # 赋值脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"