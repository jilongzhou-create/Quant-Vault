import numpy as np
import pandas as pd

class EpuPanicExhaustionPulseFactor:
    """经济政策不确定性恐慌衰竭脉冲因子 (microstructure/unstructured)

    逻辑: 采用基于非结构化新闻文本生成的美国经济政策不确定性指数(usepuindxd)。当宏观政策不确定性发生极端飙升（公众与媒体恐慌极值）时，往往伴随着市场流动性冲击和美债抛售；严格遵循二阶导数与极值衰竭铁律，只有当不确定性指数极度偏离常态（Z-Score > 2.5）且开始边际回落（低于过去3日均值）时，才确认恐慌见顶，此时输出看多美债的极短期抄底脉冲信号。
    数据: usepuindxd (基于非结构化新闻数据提取的 US Economic Policy Uncertainty Index)
    触发: usepuindxd 的 252日 Z-Score > 2.5 且 当日值 < 过去3日均值
    输出: +1.0 (极端政策恐慌见顶回落，流动性挤兑衰竭，看多美债反弹)，常态下严格保持 0.0
    """

    def __init__(self):
        self.name = 'epu_panic_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理数据缺失情况，严格返回全 0 序列
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        epu = data['usepuindxd'].ffill()
        
        # 计算长周期的 Z-Score (使用 252 个交易日作为宏观基准)
        rolling_mean = epu.rolling(window=252, min_periods=60).mean()
        rolling_std = epu.rolling(window=252, min_periods=60).std()
        
        # 避免除 0 导致无穷大
        z_score = (epu - rolling_mean) / rolling_std.replace(0, np.nan)
        
        # 计算极短期的移动平均 (平滑新闻指数的日度极度噪音，判断二阶导数方向)
        short_term_mean = epu.rolling(window=3, min_periods=2).mean()
        
        # 铁律2: 必须满足“极值 + 衰竭”的二阶导数特征，禁止接飞刀
        # 条件1: 政策不确定性指标处于极端高位
        condition_extreme = z_score > 2.5
        
        # 条件2: 恐慌开始衰竭，动量反转回落
        condition_exhaustion = epu < short_term_mean
        
        # 铁律1: 零值休眠 (Sniper Pulse)，常态下信号必须为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 仅在条件同时满足的瞬间，输出 +1.0 的狙击脉冲信号
        signal[condition_extreme & condition_exhaustion] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"