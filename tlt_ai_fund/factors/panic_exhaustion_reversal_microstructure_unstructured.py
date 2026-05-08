import numpy as np
import pandas as pd

class PanicExhaustionReversalFactor:
    """恐慌极值与衰竭反转因子 (Panic Exhaustion Reversal)

    逻辑: 在流动性危机的恐慌极值处捕捉衰竭信号，等待恐慌见顶回落后抄底美债。
          结合了股市波动率(VIX)、金市波动率(GVZ)以及金融压力指数(STLFSI4)。
          只有当指标处于极度恐慌水位且开始边际回落时，才触发脉冲。常态下严格休眠为0，避免在主跌浪中接飞刀。
    数据: vixcls, gvzcls, stlfsi4
    触发: 各个指标的 252日 Z-Score > 2.5 且 当日值 < 过去3日均值 (极值+衰竭二阶导数)
    输出: +1.0 (恐慌见顶回落, 脉冲看多美债作为防守反击), 其余时间为 0.0
    """

    def __init__(self):
        self.name = 'panic_exhaustion_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        long_cond = pd.Series(False, index=data.index)

        # 遍历代表市场恐慌与流动性压力的关键指标
        panic_indicators = ['vixcls', 'gvzcls', 'stlfsi4']
        
        for col in panic_indicators:
            if col in data.columns:
                # 前向填充缺失值以防止数据缺失导致的信号断层
                series = data[col].ffill()
                
                # 计算 252日 Z-Score，定位极端恐慌水位
                rolling_mean_252 = series.rolling(window=252, min_periods=21).mean()
                rolling_std_252 = series.rolling(window=252, min_periods=21).std().replace(0, np.nan)
                z_score = (series - rolling_mean_252) / rolling_std_252
                
                # 计算 3日均值，用于捕捉边际回落 (衰竭条件)
                short_mean_3 = series.rolling(window=3, min_periods=2).mean()
                
                # 核心铁律2: 极值 + 衰竭
                # 条件1: 必须处于极端高位 (Z-Score > 2.5)
                # 条件2: 动量必须开始反转回落 (当前值 < 过去3日均值)
                col_long = (z_score > 2.5) & (series < short_mean_3)
                
                # 只要任一领域的恐慌指标出现极值衰竭，即触发抄底脉冲
                long_cond = long_cond | col_long

        # 生成最终脉冲信号
        signal[long_cond] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"