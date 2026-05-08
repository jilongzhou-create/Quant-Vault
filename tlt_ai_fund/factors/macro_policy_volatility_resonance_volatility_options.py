import numpy as np
import pandas as pd

class MacroPolicyVolatilityResonanceFactor:
    """宏观政策与波动率共振反转因子 (volatility/microstructure)

    逻辑: 监控经济政策不确定性(EPU)与市场隐含波动率(VIX)的史诗级共振。当宏观叙事崩塌导致微观期权市场踩踏飙升时，美债具备极致避险价值；一旦共振恐慌衰竭回落，释放看多美债的脉冲。常态下输出0.0保持休眠。
    数据: usepuindxd (经济政策不确定性指数), vixcls (VIX指数)
    触发: 极值 (EPU与VIX的126日Z-Score双双>2.0) + 衰竭 (两者同步日度回落且VIX下穿3日均值)
    输出: [-1.0, 1.0] 的狙击手多空脉冲信号
    """

    def __init__(self):
        self.name = 'macro_policy_vol_resonance_reversal'
        self.window = 126
        self.long_z_threshold = 2.0
        self.short_z_threshold = -1.5

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (初始化为全 0.0)
        signal = pd.Series(0.0, index=data.index)
        
        # 必须处理数据缺失情况
        if 'usepuindxd' not in data.columns or 'vixcls' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        
        # 铁律3: 边际变化 (通过计算中长期滚动 Z-Score，捕捉水位的边际极端偏离)
        epu_mean = epu.rolling(self.window).mean()
        epu_std = epu.rolling(self.window).std()
        vix_mean = vix.rolling(self.window).mean()
        vix_std = vix.rolling(self.window).std()
        
        # 加上微小常数防止除以零的计算溢出
        epu_z = (epu - epu_mean) / (epu_std + 1e-6)
        vix_z = (vix - vix_mean) / (vix_std + 1e-6)
        
        # 铁律2: 二阶导数 (绝不接飞刀，必须等极端情绪开始瓦解衰竭)
        # --------------------------------------------------------
        # 看多信号: 政策与市场共振恐慌见顶，流动性冲击衰竭，避险资金大量买入美债
        long_extreme = (epu_z > self.long_z_threshold) & (vix_z > self.long_z_threshold)
        long_exhaustion = (epu.diff() < 0) & (vix.diff() < 0) & (vix < vix.rolling(3).mean())
        
        # 看空信号: 极度自满导致共振下行衰竭，风险偏好反转初期抛售美债
        short_extreme = (epu_z < self.short_z_threshold) & (vix_z < self.short_z_threshold)
        short_exhaustion = (epu.diff() > 0) & (vix.diff() > 0) & (vix > vix.rolling(3).mean())
        
        # 严格基于触发条件赋值，构建脉冲信号
        signal.loc[long_extreme & long_exhaustion] = 1.0
        signal.loc[short_extreme & short_exhaustion] = -1.0
        
        # 清理由于早期滚动窗口带来的潜在 NaN 
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window})"