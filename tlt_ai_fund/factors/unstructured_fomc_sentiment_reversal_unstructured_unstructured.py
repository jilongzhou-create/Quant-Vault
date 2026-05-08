import numpy as np
import pandas as pd

class UnstructuredEpuPanicPulseFactor:
    """经济政策不确定性恐慌脉冲因子 (unstructured/unstructured)

    逻辑: 当美国经济政策不确定性指数(EPU)短期内大幅飙升时, 往往伴随避险情绪升温, 利好美债避险配置。反之不确定性极速消除时压抑美债。为何是脉冲: 恐慌宣泄只在突变瞬间剧烈定价, 常态下EPU不提供收益指引。
    数据: usepuindxd (经济政策不确定性指数)
    触发: 5日变化量的63日 Z-Score > 1.5 且当日 EPU 边际回落 -> +1.0 (恐慌见顶衰竭看多美债)
          5日变化量的63日 Z-Score < -1.5 且当日 EPU 边际回升 -> -1.0 (乐观见底衰竭看空美债)
    输出: 脉冲信号 [-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'unstructured_epu_panic_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 填补可能缺失的日频数据
        epu = data['usepuindxd'].ffill()
        
        # 铁律3: 边际变化。使用5日(一周)变化量衡量预期的短期突变
        epu_diff = epu.diff(5)
        
        # 计算动量的滚动 Z-Score (63个交易日约等于一个季度)
        roll_mean = epu_diff.rolling(window=63, min_periods=21).mean()
        roll_std = epu_diff.rolling(window=63, min_periods=21).std()
        
        # 防止除零报错
        roll_std = roll_std.replace(0, np.nan)
        epu_diff_z = (epu_diff - roll_mean) / roll_std
        
        # 铁律2: 二阶导数衰竭条件。EPU极度飙升后开始边际回落, 避免接主跌浪/主升浪飞刀
        epu_daily_diff = epu.diff(1)
        
        # 触发条件: 极值突变(Z>1.5 放宽阈值确保合理Trigger率) + 边际衰竭反转
        bull_cond = (epu_diff_z > 1.5) & (epu_daily_diff < 0)
        bear_cond = (epu_diff_z < -1.5) & (epu_daily_diff > 0)
        
        signal.loc[bull_cond] = 1.0
        signal.loc[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"