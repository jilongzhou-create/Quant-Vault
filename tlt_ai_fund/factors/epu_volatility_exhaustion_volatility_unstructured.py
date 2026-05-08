import numpy as np
import pandas as pd

class EpuVolatilityExhaustionFactor:
    """经济政策不确定性与波动率共振衰竭因子 (volatility/unstructured)

    逻辑: 当基于新闻的经济政策不确定性(EPU)与VIX同时狂飙至极端水平并开始回落时，表明流动性恐慌见顶，避险情绪反转或央行救市预期将推动美债大幅反弹。反之，当市场极度自满开始瓦解时，看空美债。
    数据: usepuindxd (经济政策不确定性指数), vixcls (VIX波动率指数)
    触发: 联合恐慌指标 Z-Score > 2.5 且 跌破3日均值 且 VIX当日回落 -> +1.0 脉冲做多；联合 Z-Score < -2.0 且 升破3日均值 且 VIX当日上升 -> -1.0 脉冲做空。
    输出: 严格的狙击手级脉冲信号 [-1.0, 1.0]
    """

    def __init__(self, window=63, extreme_high=2.5, extreme_low=-2.0):
        self.name = 'epu_volatility_exhaustion'
        self.window = window
        self.extreme_high = extreme_high
        self.extreme_low = extreme_low

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 初始信号必须为 0.0 (零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查数据完备性
        if 'usepuindxd' not in data.columns or 'vixcls' not in data.columns:
            return signal
            
        # 前向填充处理缺失值
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        
        # 铁律3: 边际变化，计算基于窗口的动态 Z-Score
        epu_std = epu.rolling(self.window).std().replace(0, 1e-5) # 防除零
        epu_z = (epu - epu.rolling(self.window).mean()) / epu_std
        
        vix_std = vix.rolling(self.window).std().replace(0, 1e-5)
        vix_z = (vix - vix.rolling(self.window).mean()) / vix_std
        
        # 结合跨资产恐慌共振的综合得分
        panic_score = epu_z + vix_z
        
        # 铁律2: 二阶导数 (极端高位 + 开始回落确认)
        # 多头条件：极度恐慌见顶瓦解 -> 资金涌入美债避险或押注联储转鸽
        cond_long_extreme = panic_score > self.extreme_high
        cond_long_exhaust = panic_score < panic_score.rolling(3).mean()
        cond_long_confirm = vix.diff() < 0
        
        # 空头条件：极度自满见底回升 -> 风险偏好过高、宽松预期退潮、美债被抛售
        cond_short_extreme = panic_score < self.extreme_low
        cond_short_exhaust = panic_score > panic_score.rolling(3).mean()
        cond_short_confirm = vix.diff() > 0
        
        # 生成脉冲触发信号
        signal[cond_long_extreme & cond_long_exhaust & cond_long_confirm] = 1.0
        signal[cond_short_extreme & cond_short_exhaust & cond_short_confirm] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, extreme_high={self.extreme_high}, extreme_low={self.extreme_low})"