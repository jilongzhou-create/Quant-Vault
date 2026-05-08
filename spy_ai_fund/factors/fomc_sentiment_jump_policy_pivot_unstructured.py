import numpy as np
import pandas as pd

class FomcSentimentJumpFactor:
    """FOMC情绪边际突变脉冲因子 (policy_pivot/unstructured)

    逻辑: 美联储货币政策预期的边际变化(动量)决定了流动性冲量的方向。当FOMC声明的NLP情感得分发生鸽派突变(跳升)时，意味着紧缩周期衰竭或宽松周期开启，引爆风险资产做多情绪；反之鹰派剧烈突降则构成短期抛压。
    数据: [fomc_sentiment]
    输出: [+1.0=强烈看多(鸽派突变), -1.0=看空(鹰派突变)]
    触发条件: fomc_sentiment单日边际差分 > +0.2 或 < -0.2 瞬间触发，信号维持3到5个交易日，预期 Trigger Rate 5%-10%
    """

    def __init__(self, dovish_threshold: float = 0.2, hawkish_threshold: float = -0.2, dovish_pulse_days: int = 5, hawkish_pulse_days: int = 3):
        self.name = 'fomc_sentiment_jump'
        self.dovish_threshold = dovish_threshold
        self.hawkish_threshold = hawkish_threshold
        # 美股长牛属性: 政策转向看多脉冲可持续发酵，但看空(逆势)脉冲必须极短以防踏空主升浪
        self.dovish_pulse_days = dovish_pulse_days
        self.hawkish_pulse_days = hawkish_pulse_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        sentiment = data['fomc_sentiment']
        
        # 边际变化铁律: 绝对禁止直接输出低频阶梯数据的绝对值，必须使用.diff()计算动量跳跃
        sentiment_change = sentiment.diff()
        
        # 捕捉预期剧变的瞬间 (Pulse Triggers)
        is_dovish_jump = sentiment_change > self.dovish_threshold
        is_hawkish_jump = sentiment_change < self.hawkish_threshold
        
        # 将单一瞬间跳跃转化为极短的脉冲信号 (狙击手级)
        # rolling.max() 将触发当天的1.0顺延指定的极端交易日窗口
        dovish_pulse = is_dovish_jump.rolling(window=self.dovish_pulse_days, min_periods=1).max().fillna(0)
        hawkish_pulse = is_hawkish_jump.rolling(window=self.hawkish_pulse_days, min_periods=1).max().fillna(0)
        
        # 写入信号
        signal.loc[dovish_pulse == 1] = 1.0
        signal.loc[hawkish_pulse == 1] = -1.0
        
        # 确保严格遵守 [-1.0, 1.0] 输出范围约束
        signal = signal.clip(-1.0, 1.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(dovish_threshold={self.dovish_threshold}, hawkish_threshold={self.hawkish_threshold})"