import numpy as np
import pandas as pd

class VolatilityUncertaintyReversalFactor:
    """Volatility and Uncertainty Reversal Factor (volatility/nonlinear)

    逻辑: 跨市场恐慌（VIX）与经济政策不确定性（USEPUINDXD）同时飙升至极值代表系统性压力或危机，随后两者同步回落说明恐慌开始衰竭（流动性危机缓解或政策干预生效），此时是做多避险资产美债（TLT）的最佳脉冲时点；反之，极度自满且波动率抬头时做空。严格遵守二阶导数和零值休眠铁律。
    数据: vixcls, usepuindxd
    触发: VIX 252日 Z-Score > 2.0 且 USEPU 252日 Z-Score > 1.5，且双双跌破短期均线并边际回落时输出 +1.0；极度低迷且边际反弹时输出 -1.0。
    输出: [-1.0, 1.0] 狙击手级别的脉冲信号。常态为 0.0。
    """

    def __init__(self):
        self.name = 'volatility_uncertainty_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'usepuindxd']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 1. 基础数据前向填充，防止日期间隙产生 NaN
        vix = data['vixcls'].ffill()
        epu_raw = data['usepuindxd'].ffill()
        
        # 政策不确定性数据日频噪音极大，使用5日均值平滑以获取真实趋势
        epu = epu_raw.rolling(window=5, min_periods=1).mean()
        
        # 2. 计算具有宏观经济学含义的 252日(1年) 滚动 Z-Score
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std()
        vix_z = (vix - vix_mean) / vix_std
        
        epu_mean = epu.rolling(window=252, min_periods=60).mean()
        epu_std = epu.rolling(window=252, min_periods=60).std()
        epu_z = (epu - epu_mean) / epu_std
        
        # 3. 二阶导数与边际变化 (动量衰竭条件，绝对禁止接飞刀)
        # 多头条件：跌破3日/5日均线且动量为负
        vix_exhausted = (vix < vix.rolling(window=3, min_periods=1).mean()) & (vix.diff() < 0)
        epu_exhausted = (epu < epu.rolling(window=5, min_periods=1).mean()) & (epu.diff() < 0)
        
        # 空头条件：突破3日/5日均线且动量为正
        vix_spiking = (vix > vix.rolling(window=3, min_periods=1).mean()) & (vix.diff() > 0)
        epu_spiking = (epu > epu.rolling(window=5, min_periods=1).mean()) & (epu.diff() > 0)
        
        # 4. 多空信号综合触发
        # 恐慌极值且开始瓦解 -> 衰竭买入脉冲 (+1.0)
        long_trigger = (vix_z > 2.0) & (epu_z > 1.5) & vix_exhausted & epu_exhausted
        
        # 极度自满且突然发散 -> 抛售风险资产与长债的脉冲 (-1.0)
        short_trigger = (vix_z < -1.2) & (epu_z < -1.2) & vix_spiking & epu_spiking
        
        signal.loc[long_trigger] = 1.0
        signal.loc[short_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"