import numpy as np
import pandas as pd

class VolatilityMicrostructureExhaustionFactor:
    """Volatility Microstructure Exhaustion (microstructure/options)

    逻辑: 捕捉期权隐含波动率的极值反转。恐慌极值回落时，流动性抛售消退，实质性避险资金回流美债（脉冲做多）；极度低波动率被打破时，风险平价基金降杠杆导致股债双杀抛压（脉冲做空）。
    数据: vixcls (标普500期权隐含波动率)
    触发: VIX 63日 Z-Score > 2.0 且当日低于3日均值并下跌（看多）；Z-Score < -1.5 且当日高于3日均值并上涨（看空）。
    输出: +1.0 (恐慌衰竭, 买入美债), -1.0 (平庸被打破, 抛售美债), 常态为 0.0
    """

    def __init__(self):
        self.name = 'volatility_microstructure_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns:
            return signal
            
        # 提取期权隐含波动率数据并前向填充
        vix = data['vixcls'].ffill()
        
        # 使用63个交易日(约一季度)作为微观结构情绪波动的基准窗口
        # 避免使用固定绝对值，而是使用滚动相对水位 (边际变化铁律)
        roll_mean = vix.rolling(window=63).mean()
        roll_std = vix.rolling(window=63).std()
        
        # 避免除以零导致无穷大
        roll_std = roll_std.replace(0.0, np.nan)
        zscore = (vix - roll_mean) / roll_std
        
        # 极短期的平滑均线，用于判断边际衰竭
        vix_ma3 = vix.rolling(window=3).mean()
        
        # ---------------------------------------------------------------------
        # 多头脉冲: 极度恐慌衰竭
        # 绝对禁止 VIX > 2.0 直接买入 (避免接飞刀)!
        # 必须满足二阶导数铁律: 极端高位 + 跌破3日均值 + 动量转负
        # ---------------------------------------------------------------------
        long_cond = (zscore > 2.0) & (vix < vix_ma3) & (vix.diff() < 0)
        
        # ---------------------------------------------------------------------
        # 空头脉冲: 极度乐观惊醒 (波动率被突然做多)
        # 长期的低波动率(平庸期)被打破瞬间，风险平价等策略抛售股债
        # 极低水平 + 升破3日均值 + 动量转正
        # ---------------------------------------------------------------------
        short_cond = (zscore < -1.5) & (vix > vix_ma3) & (vix.diff() > 0)
        
        # 脉冲触发 (零值休眠铁律: 其余时间均保持 0.0)
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"