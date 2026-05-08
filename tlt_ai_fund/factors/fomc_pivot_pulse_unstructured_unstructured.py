import numpy as np
import pandas as pd

class FomcPivotPulseFactor:
    """政策预期突变脉冲因子 (unstructured/NLP Sentiment)

    逻辑: 捕捉美联储FOMC声明情绪(鹰鸽得分)的极端边际变化，并利用对政策最敏感的2年期美债收益率(dgs2)的回落作为顺势确认，规避“接飞刀”，在市场真正Price-in且预期反转的瞬间产生狙击手级脉冲信号。
    数据: fomc_sentiment (FOMC鹰鸽情绪得分), dgs2 (2年期美债收益率)
    触发: FOMC情绪5日变化量的252日 Z-Score > 2.5 (预期极端突变) 且 dgs2 的3日动量 < 0 (前端利率顺势回落确认)
    输出: 仅在突变发生且由利率动量确认的极短窗口内输出 +1.0 (看多) / -1.0 (看空)，常态严格为 0.0。
    """

    def __init__(self, sentiment_window=5, zscore_window=252, z_threshold=2.5, conf_window=3):
        self.name = 'fomc_pivot_pulse'
        self.sentiment_window = sentiment_window
        self.zscore_window = zscore_window
        self.z_threshold = z_threshold
        self.conf_window = conf_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号设为 0.0 (铁律1: 零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        # 必须处理数据缺失的情况
        if 'fomc_sentiment' not in data.columns or 'dgs2' not in data.columns:
            return signal
            
        # 铁律3: 边际变化 (Marginal Change Only)
        # 禁止使用绝对值，通过差分捕捉 FOMC 情绪得分的边际突变 (鸽派突变 > 0, 鹰派突变 < 0)
        sent_diff = data['fomc_sentiment'].diff(self.sentiment_window)
        
        # 计算情绪变化量的 Z-Score (反映政策偏离常态预期的极值程度)
        sent_mean = sent_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).mean()
        sent_std = sent_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).std()
        sent_zscore = (sent_diff - sent_mean) / (sent_std + 1e-6)
        
        # 铁律2: 二阶导数/衰竭确认 (Anti-Catch-Falling-Knife)
        # 使用对政策预期最敏感的前瞻指标 dgs2 (2年期收益率) 的动量进行过滤
        # 如果情绪突变但收益率背离，说明市场在抵抗(死于主跌浪)，必须等收益率同向配合才触发信号
        dgs2_momentum = data['dgs2'].diff(self.conf_window)
        
        # 触发条件: 鸽派情绪突变极值 + 2年期美债收益率衰竭下行 -> 脉冲看多美债
        bull_cond = (sent_zscore > self.z_threshold) & (dgs2_momentum < 0)
        
        # 触发条件: 鹰派情绪突变极值 + 2年期美债收益率突破上行 -> 脉冲看空美债
        bear_cond = (sent_zscore < -self.z_threshold) & (dgs2_momentum > 0)
        
        # 赋值极端事件当天的脉冲信号
        signal[bull_cond] = 1.0
        signal[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(sentiment_window={self.sentiment_window}, zscore_window={self.zscore_window}, z_threshold={self.z_threshold}, conf_window={self.conf_window})"