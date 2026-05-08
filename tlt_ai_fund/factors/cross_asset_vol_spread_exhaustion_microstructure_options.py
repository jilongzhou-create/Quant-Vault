import numpy as np
import pandas as pd

class OptionsCrossVolDivergenceFactor:
    """期权跨资产波动率背离衰竭因子 (Microstructure / Options)

    逻辑: 捕捉期权微观结构中跨资产波动率(VIX与黄金GVZ)在季度期权合约周期(63日)内的极端背离与衰竭。
          当VIX远超GVZ(股市恐慌主导)并开始衰竭时，避险资金将流出美债追逐风险资产(Risk-On)，对应看空美债(-1.0);
          当GVZ远超VIX(通胀/法币信用恐慌主导)并开始衰竭时，实际利率冲击见顶，对应抄底美债(+1.0)。
          该脉冲信号精准避免单边趋势中的接飞刀，且信号极性与跨资产宏观底仓流向严格对齐，解决与其他因子的同向摩擦(Toxic)问题。
    数据: vixcls, gvzcls
    触发: VIX-GVZ差值的63日Z-Score绝对值 > 1.5 (匹配5%-15%的目标触发率), 且差值跌破/突破3日均线并出现反转拐点。
    输出: +1.0 (通胀恐慌衰竭，看多美债), -1.0 (增长恐慌衰竭，看空美债), 0.0 (常态休眠)
    """

    def __init__(self):
        self.name = 'options_cross_vol_divergence'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值休眠信号 (铁律1)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需字段是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产波动率微观结构差值 (Equity Vol vs Gold Vol)
        vol_spread = vix - gvz
        
        # 计算63日(适应期权季度展期周期)滚动Z-Score，寻找局部极值
        roll_mean = vol_spread.rolling(window=63, min_periods=21).mean()
        roll_std = vol_spread.rolling(window=63, min_periods=21).std()
        
        spread_z = (vol_spread - roll_mean) / (roll_std + 1e-8)
        
        # 边际变化与二阶导数衰竭条件 (铁律2 & 铁律3)
        spread_ma3 = vol_spread.rolling(window=3).mean()
        spread_diff = vol_spread.diff(1)
        
        # 条件1: 通胀/信用恐慌衰竭 (GVZ极端飙升后见顶回落) -> 实际利率冲击见顶 -> 做多美债 (+1.0)
        # 逻辑: Z-Score极负(极值) + 差值开始上升(二阶反转/边际变化)
        long_cond = (spread_z < -1.5) & (vol_spread > spread_ma3) & (spread_diff > 0)
        
        # 条件2: 股市/增长恐慌衰竭 (VIX极端飙升后见顶回落) -> 避险资金撤出债券/Risk-On -> 做空美债 (-1.0)
        # 逻辑: Z-Score极正(极值) + 差值开始下降(二阶反转/边际变化)
        short_cond = (spread_z > 1.5) & (vol_spread < spread_ma3) & (spread_diff < 0)
        
        # 只在触发条件满足时赋值为脉冲
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"