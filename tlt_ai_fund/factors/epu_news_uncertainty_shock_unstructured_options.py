import numpy as np
import pandas as pd

class EpuNewsUncertaintyShockFactor:
    """新闻政策不确定性极端反转因子 (unstructured/options)

    逻辑: USEPUINDXD (基于非结构化新闻文本的每日经济政策不确定性指数) 在宏观叙事中扮演着“政策隐含波动率”的角色。当不确定性指数发生极端飙升(边际变化极大)且开始见顶回落时，标志着宏观恐慌情绪的极值已过，美联储往往已经出手安抚市场，避险资金和宽松预期双重共振，生成看多美债(TLT)的脉冲。反之，当极端自满被打破时，预示紧缩或通胀风险重燃，生成看空脉冲。
    数据: usepuindxd (Daily Economic Policy Uncertainty Index, 基于海量新闻文本挖掘)
    触发: 5日变化量的 252日 Z-Score > 2.5 且 当前绝对值 < 3日均值(恐慌衰竭) -> +1.0 看多；Z-Score < -2.5 且 当前绝对值 > 3日均值(自满破裂) -> -1.0 看空
    输出: 严格控制在 [-1.0, 1.0] 的极值衰竭脉冲信号 (Sniper Pulse)
    """

    def __init__(self, diff_days: int = 5, window: int = 252, smooth_days: int = 3, z_thresh: float = 2.5):
        self.name = 'epu_news_uncertainty_shock'
        # diff_days: 5个交易日(单周)，代表市场消化政策突变的一个标准发酵周期
        self.diff_days = diff_days
        # window: 252个交易日(1年)，代表宏观周期的基准回望长度
        self.window = window
        # smooth_days: 3个交易日，用于极短期二阶导数(趋势反转)的衰竭确认
        self.smooth_days = smooth_days
        # z_thresh: 2.5，统计学上正态分布约前 0.6% 的极端尾部概率
        self.z_thresh = z_thresh

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理数据缺失的情况 (返回全 0 Series)
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        epu = data['usepuindxd'].ffill()
        
        # =================================================================
        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝不直接使用 EPU 水平值，必须使用5日(单周)变化量，捕捉预期突变瞬间
        # =================================================================
        epu_diff = epu.diff(self.diff_days)
        
        # 计算基于 252 日滚动窗口的 Z-Score (衡量异动极端程度)
        roll_mean = epu_diff.rolling(window=self.window, min_periods=self.window//2).mean()
        roll_std = epu_diff.rolling(window=self.window, min_periods=self.window//2).std()
        roll_std = roll_std.replace(0, np.nan) # 防止除零导致 inf
        
        zscore = (epu_diff - roll_mean) / roll_std
        
        # =================================================================
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 绝对禁止 "只要恐慌指标极高就买入"，必须叠加动能衰竭条件
        # =================================================================
        epu_ma3 = epu.rolling(window=self.smooth_days).mean()
        exhaustion_high = epu < epu_ma3  # 波动率回落: 极端恐慌开始消退
        exhaustion_low = epu > epu_ma3   # 波动率抬头: 极端自满开始被打破
        
        # =================================================================
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 非触发日严格休眠为 0.0，目标达成短线狙击式打击
        # =================================================================
        signal = pd.Series(0.0, index=data.index)
        
        # 极值条件 + 衰竭条件，两者必须同时满足
        long_cond = (zscore > self.z_thresh) & exhaustion_high
        short_cond = (zscore < -self.z_thresh) & exhaustion_low
        
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(diff_days={self.diff_days}, window={self.window}, z_thresh={self.z_thresh})"