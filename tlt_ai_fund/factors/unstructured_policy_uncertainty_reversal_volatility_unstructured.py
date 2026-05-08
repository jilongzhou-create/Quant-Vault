import numpy as np
import pandas as pd

class UnstructuredPolicyUncertaintyReversalFactor:
    """经济政策不确定性反转脉冲因子 (volatility/unstructured)

    逻辑: USEPUINDXD (经济政策不确定性指数) 是基于新闻文本的非结构化宏观波动率指标。当政策恐慌达到极端高位并开始回落时，代表政策紧缩预期见顶或不确定性消退，避险情绪衰竭，利多美债；当市场极度自满(不确定性极低)且波动率突然抬头时，往往预示政策意外或抛售风险，利空美债。采用严格的二阶导数过滤以规避主跌浪，形成狙击手脉冲。
    数据: usepuindxd (经济政策不确定性指数), vixcls (VIX波动率指数)
    触发: 多头 = EPU 126日Z-Score > 2.5 且 EPU及VIX同步跌破5日均线；空头 = EPU 126日Z-Score < -2.0 且 EPU及VIX同步突破5日均线
    输出: +1.0 政策恐慌消退(买入美债)，-1.0 极度自满被打破(做空美债)，其余为 0.0
    """

    def __init__(self):
        self.name = 'unstructured_policy_uncertainty_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 - 初始全 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 数据依赖检查
        if 'usepuindxd' not in data.columns or 'vixcls' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        
        # 计算 126日 (半年) Z-Score 衡量中期极端情绪水位
        epu_mean = epu.rolling(window=126).mean()
        epu_std = epu.rolling(window=126).std()
        
        # 避免分母为 0 导致的数据异常
        epu_zscore = (epu - epu_mean) / epu_std.replace(0, np.nan)
        
        # 铁律2 & 3: 二阶导数与边际变化 - 衰竭与反转的动量确认
        epu_ma5 = epu.rolling(window=5).mean()
        vix_ma5 = vix.rolling(window=5).mean()
        
        # 衰竭条件: 当日边际下降 且 跌破近期均线
        epu_falling = (epu < epu_ma5) & (epu.diff() < 0)
        vix_falling = (vix < vix_ma5) & (vix.diff() < 0)
        
        # 爆发条件: 当日边际上升 且 突破近期均线
        epu_rising = (epu > epu_ma5) & (epu.diff() > 0)
        vix_rising = (vix > vix_ma5) & (vix.diff() > 0)
        
        # 触发脉冲信号 (Sniper Pulse)
        # 做多脉冲: 政策恐慌处于极值 (Z > 2.5) + EPU开始回落 + VIX跨资产确认同步回落
        long_cond = (epu_zscore > 2.5) & epu_falling & vix_falling
        
        # 做空脉冲: 极度自满 (Z < -2.0) + EPU意外飙升 + VIX跨资产确认同步飙升
        short_cond = (epu_zscore < -2.0) & epu_rising & vix_rising
        
        # 赋值非零脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"