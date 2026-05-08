import numpy as np
import pandas as pd

class FomcSentimentPulseFactor:
    """FOMC Sentiment Marginal Pulse (policy_pivot/unstructured)

    逻辑: 分析基于NLP提取的FOMC声明情绪得分变化(边际预期突变)。鸽派转向(预期放松)带来流动性支撑, 产生看多脉冲(+1.0)。轻微的鹰派突变引发轻度恐慌与趋势恶化, 产生看空脉冲(-1.0)。当出现极端的鹰派休克时, 市场会产生极度恐慌(接飞刀风险); 根据标普500均值回归的长牛特性, 此时强制等待2个交易日让恐慌衰竭, 然后再输出强烈的抄底看多脉冲(+1.0)。
    数据: fomc_sentiment
    输出: 脉冲信号 [-1.0, 1.0]
    触发条件: 情绪得分出现跃升(鸽派)或跨越0轴触发看多; 小幅下跌触发看空; 极度下跌则延迟触发看多。脉冲保持3天以确保 Trigger Rate 落在 5%-15% 之间。
    """

    def __init__(self, dove_thresh: float = 0.15, hawk_mild_thresh: float = -0.15, hawk_extreme_thresh: float = -0.40):
        self.name = 'fomc_sentiment_pulse'
        self.dove_thresh = dove_thresh
        self.hawk_mild_thresh = hawk_mild_thresh
        self.hawk_extreme_thresh = hawk_extreme_thresh

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # FOMC 情绪得分是低频阶梯状数据, 严格遵循边际变化铁律, 只能使用 .diff() 捕捉预期反转瞬间
        fomc = data['fomc_sentiment'].ffill()
        fomc_diff = fomc.diff()
        
        # 1. 鸽派突变 (Dovish Pivot) -> 直接看多
        # 满足条件: 情绪得分跳升超过阈值, 或者由负转正(预期实质性反转)
        dovish = (fomc_diff >= self.dove_thresh) | ((fomc > 0) & (fomc.shift(1) < 0))
        
        # 2. 轻度鹰派突变 (Mild Hawkish) -> 趋势恶化, 轻度恐慌 -> 看空
        mild_hawk = (fomc_diff <= self.hawk_mild_thresh) & (fomc_diff > self.hawk_extreme_thresh)
        
        # 3. 极端鹰派休克 (Extreme Hawkish) -> 极度恐慌
        extreme_hawk = (fomc_diff <= self.hawk_extreme_thresh)
        
        # 扩展脉冲宽度至3个交易日, 避免单日信号过窄导致 Trigger Rate 达不到 5% 下限
        buy_dove = dovish.rolling(window=3, min_periods=1).max() > 0
        sell_hawk = mild_hawk.rolling(window=3, min_periods=1).max() > 0
        
        # 二阶导数与防接飞刀铁律: 极端恐慌下绝对禁止立刻买入! 必须等恐慌开始衰竭
        # 延迟2个交易日(避开主跌浪), 然后再发出为期3天的抄底看多脉冲
        extreme_exhausted = extreme_hawk.shift(2).fillna(False)
        buy_extreme = extreme_exhausted.rolling(window=3, min_periods=1).max() > 0
        
        # 信号赋值
        signal.loc[sell_hawk] = -1.0
        signal.loc[buy_dove] = 1.0
        signal.loc[buy_extreme] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(dove_thresh={self.dove_thresh}, hawk_mild={self.hawk_mild_thresh}, hawk_extreme={self.hawk_extreme_thresh})"