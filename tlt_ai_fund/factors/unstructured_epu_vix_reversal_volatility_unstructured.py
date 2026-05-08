import numpy as np
import pandas as pd

class UnstructuredEpuVixReversalFactor:
    """经济政策不确定性与跨资产波动率共振衰竭因子 (volatility/unstructured)

    逻辑: 结合基于新闻文本的非结构化指标(USEPUINDXD)与跨资产波动率(VIX)构建复合冲击指数。当政策不确定性引发跨资产恐慌并狂飙至极值时(复合Z-Score>3.5)，债券常因流动性冲击被错杀；一旦两者同步出现边际衰竭（从极值跌破3日均线且动量向下），标志着宏观恐慌瓦解，避险和流动性资金重归美债，触发强劲的看多脉冲。反之，极度自满后的波动率抬头触发看空。
    数据: usepuindxd (美国经济政策不确定性指数), vixcls (VIX波动率)
    触发: 复合 Z-Score > 3.5 且两者同步回落(二阶衰竭) -> +1.0; 复合 Z-Score < -2.5 且两者同步抬头 -> -1.0
    输出: [-1.0, 1.0] 的狙击手级脉冲信号
    """

    def __init__(self, window=252, ma_window=3, composite_upper=3.5, composite_lower=-2.5):
        self.name = 'unstructured_epu_vix_reversal'
        self.window = window
        self.ma_window = ma_window
        self.composite_upper = composite_upper
        self.composite_lower = composite_lower

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (默认全为 0.0)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需数据列是否存在
        required_cols = ['usepuindxd', 'vixcls']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        # 提取并前向填充缺失数据
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        
        # 避免前瞻偏差：使用过去 252 日数据计算 Z-Score
        epu_mean = epu.rolling(window=self.window, min_periods=self.window//2).mean()
        epu_std = epu.rolling(window=self.window, min_periods=self.window//2).std()
        epu_z = (epu - epu_mean) / epu_std.replace(0, np.nan)
        
        vix_mean = vix.rolling(window=self.window, min_periods=self.window//2).mean()
        vix_std = vix.rolling(window=self.window, min_periods=self.window//2).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        # 构建复合的非结构化政策与市场恐慌冲击指数
        composite_shock = epu_z + vix_z
        
        # 铁律2 & 3: 二阶导数与边际变化条件 (绝对禁止极值直接买入，防接飞刀)
        # 衰竭确认: 当日水平 < 近3日均线 且 边际差分为负(正在下行)
        epu_exhaustion = (epu < epu.rolling(window=self.ma_window).mean()) & (epu.diff() < 0)
        vix_exhaustion = (vix < vix.rolling(window=self.ma_window).mean()) & (vix.diff() < 0)
        
        # 抬头确认: 当日水平 > 近3日均线 且 边际差分为正(正在上行)
        epu_rising = (epu > epu.rolling(window=self.ma_window).mean()) & (epu.diff() > 0)
        vix_rising = (vix > vix.rolling(window=self.ma_window).mean()) & (vix.diff() > 0)
        
        # 触发看多脉冲: 宏观恐慌极值 + 跨资产同步衰竭
        long_condition = (composite_shock > self.composite_upper) & epu_exhaustion & vix_exhaustion
        
        # 触发看空脉冲: 宏观极度自满 + 波动率边际同步抬头
        short_condition = (composite_shock < self.composite_lower) & epu_rising & vix_rising
        
        # 只在触发条件瞬间输出 1.0 或 -1.0 信号
        signal.loc[long_condition] = 1.0
        signal.loc[short_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, composite_upper={self.composite_upper}, composite_lower={self.composite_lower})"