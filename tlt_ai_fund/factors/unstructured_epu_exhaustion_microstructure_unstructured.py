import numpy as np
import pandas as pd

class UnstructuredEpuExhaustionFactor:
    """经济政策不确定性情绪衰竭因子 (microstructure/unstructured)

    逻辑: 基于新闻文本的经济政策不确定性指数(usepuindxd)捕捉市场宏观情绪的微观结构跳跃。当政策不确定性极度飙升时，市场陷入无序恐慌抛售；为避免接飞刀，必须等待恐慌情绪见顶衰竭（跌破3日均线），此时确认避险情绪落地，资金将大规模涌入长端美债，触发看多脉冲。反之亦然。常态下输出0.0。
    数据: usepuindxd (Economic Policy Uncertainty Index)
    触发: 多头 = EPU 252日 Z-Score > 2.5 AND EPU < 3日均值；空头 = EPU 252日 Z-Score < -2.0 AND EPU > 3日均值
    输出: 严格脉冲型信号, +1.0 看多 TLT (避险资金涌入), -1.0 看空 TLT (风险偏好反转), 其余 0.0
    """

    def __init__(self, zscore_window: int = 252, exhaust_window: int = 3):
        self.name = 'unstructured_epu_exhaustion_pulse'
        self.zscore_window = zscore_window
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 获取每日的非结构化情绪数据并前向填充缺失值
        epu = data['usepuindxd'].ffill()
        
        # 避免全NaN导致报错
        if epu.isna().all():
            return signal
            
        # 计算历史滚动极值 (Z-Score)
        epu_mean = epu.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).mean()
        epu_std = epu.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).std()
        
        # 避免除以0
        epu_std = epu_std.replace(0, np.nan)
        epu_zscore = (epu - epu_mean) / epu_std
        
        # 二阶导数条件: 短期动量衰竭 (当前值与短期均值的关系)
        epu_ma_short = epu.rolling(window=self.exhaust_window, min_periods=2).mean()
        
        # 边际变化条件: 必须具有日度上的真实回落动作
        epu_diff = epu.diff()
        
        # 多头信号条件 (看多美债 TLT):
        # 1. 政策不确定性在极端高位 (恐慌极值 Z > 2.5)
        # 2. 不确定性开始衰竭回落 (跌破3日均线 且 当日出现下降) -> 防止接飞刀
        long_cond = (epu_zscore > 2.5) & (epu < epu_ma_short) & (epu_diff < 0)
        
        # 空头信号条件 (看空美债 TLT):
        # 1. 政策不确定性在极端低位 (过度乐观自满 Z < -2.0)
        # 2. 不确定性开始触底回升 (升破3日均线 且 当日出现上升) -> 风险偏好见顶，抛售美债
        short_cond = (epu_zscore < -2.0) & (epu > epu_ma_short) & (epu_diff > 0)
        
        # 赋值狙击手脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, exhaust_window={self.exhaust_window})"