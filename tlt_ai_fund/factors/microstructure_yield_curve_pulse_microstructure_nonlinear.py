import numpy as np
import pandas as pd

class GlobalLiquidityExhaustionFactor:
    """全球流动性挤兑衰竭因子 (microstructure/nonlinear)

    逻辑: 结合VIX(情绪恐慌)与广义美元指数(全球流动性枯竭)构建复合压力极值。
          当市场处于极端恐慌且面临流动性挤兑时(VIX与USD同步飙升), 债券往往因去杠杆被无差别抛售(主跌浪)。
          当该双重压力见顶回落(边际衰竭)时, 强制平仓期结束, 避险资金重返美债, 触发做多脉冲。
          反之, 当处于极度自满(两者Z-score极低)且开始抬头时, 暗示紧缩周期或滞胀风险抬头, 触发做空脉冲。
    数据: vixcls (VIX波动率), dtwexbgs (广义美元指数)
    触发: (VIX 63日 Z-Score + USD 63日 Z-Score) > 1.5 且双双下穿3日均线 -> +1.0
          (VIX 63日 Z-Score + USD 63日 Z-Score) < -1.5 且双双上穿3日均线 -> -1.0
    输出: 脉冲型信号, 捕捉流动性极值的反转瞬间。
    """

    def __init__(self):
        self.name = 'global_liquidity_exhaustion_pulse'
        self.window = 63  # 1季度交易日，用于捕捉局部宏观周期极值
        self.smooth = 3   # 3日均线用于确认边际衰竭

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns or 'dtwexbgs' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        usd = data['dtwexbgs'].ffill()
        
        # 计算 63 日(一季度) Z-Score
        vix_mean = vix.rolling(self.window).mean()
        vix_std = vix.rolling(self.window).std().replace(0, 1e-5)
        vix_z = (vix - vix_mean) / vix_std
        
        usd_mean = usd.rolling(self.window).mean()
        usd_std = usd.rolling(self.window).std().replace(0, 1e-5)
        usd_z = (usd - usd_mean) / usd_std
        
        # 构建复合流动性压力指数
        comp_stress = vix_z + usd_z
        
        # 计算 3 日均线用于判断衰竭/抬头 (二阶导数铁律)
        vix_sma = vix.rolling(self.smooth).mean()
        usd_sma = usd.rolling(self.smooth).mean()
        
        vix_falling = vix < vix_sma
        usd_falling = usd < usd_sma
        
        vix_rising = vix > vix_sma
        usd_rising = usd > usd_sma
        
        # 触发条件1: 恐慌衰竭 -> 多头脉冲 (+1.0)
        # 复合压力处于极高位, 且两者都开始回落 (二阶导数衰竭)
        long_pulse = (comp_stress > 1.5) & vix_falling & usd_falling
        
        # 触发条件2: 自满反转 -> 空头脉冲 (-1.0)
        # 复合压力处于极低位(市场极度乐观), 且两者都开始抬头 (边际恶化)
        short_pulse = (comp_stress < -1.5) & vix_rising & usd_rising
        
        # 赋值信号 (严格零值休眠)
        signal[long_pulse] = 1.0
        signal[short_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, smooth={self.smooth})"