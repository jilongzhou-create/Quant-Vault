import numpy as np
import pandas as pd

class UnstructuredFomcSentimentShockFactor:
    """FOMC情绪动量突变衰竭因子 (NLP Sentiment Exhaustion)

    逻辑: 央行FOMC声明的NLP鹰鸽情绪得分为阶梯状低频数据。为了捕捉其边际变化并严格
          满足极值+衰竭二阶导铁律，本因子首先对情绪得分进行指数平滑(连续化市场消化过程)，
          随后计算其5日动量。当市场向极度鸽派突变且动能见顶回落时(美联储意外放水落地并被完全定价)，
          触发脉冲信号看多美债(避免在市场还未Price-in完成前抢跑)。相反，向极度鹰派突变并衰竭时看空。
    数据: fomc_sentiment (非结构化领域NLP央行情绪得分)
    触发: 动量 Z-Score > 2.5 (极度突变转鸽) 且动量开始衰竭(< 3日均值) -> 看多脉冲 (+1.0)
          动量 Z-Score < -2.5 (极度突变转鹰) 且动量开始衰竭(> 3日均值) -> 看空脉冲 (-1.0)
    输出: 严格遵循三大铁律的狙击手级别非连续脉冲信号，常态休眠为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_sentiment_shock_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # 填充低频阶梯数据
        fomc = data['fomc_sentiment'].ffill()
        
        # 将阶梯数据连续化，刻画政策预期在市场中发酵与传导的平滑过程
        smoothed_fomc = fomc.ewm(span=10, min_periods=5).mean()
        
        # 铁律3: 边际变化 Only (严禁使用情绪绝对水位)
        fomc_mom = smoothed_fomc.diff(5)
        
        # 计算 252 日滚动的 Z-Score
        roll_mean = fomc_mom.rolling(window=252, min_periods=60).mean()
        roll_std = fomc_mom.rolling(window=252, min_periods=60).std()
        zscore = (fomc_mom - roll_mean) / (roll_std + 1e-8)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 用 3 日均线捕捉动能的见顶回落点
        mom_ma3 = fomc_mom.rolling(window=3).mean()
        
        # 多头触发: 极度转鸽 (Z > 2.5) + 转鸽动能见顶衰竭
        dove_exhaustion = (zscore > 2.5) & (fomc_mom < mom_ma3)
        
        # 空头触发: 极度转鹰 (Z < -2.5) + 转鹰动能见底反弹衰竭
        hawk_exhaustion = (zscore < -2.5) & (fomc_mom > mom_ma3)
        
        # 铁律1: 零值休眠，脉冲输出
        signal[dove_exhaustion] = 1.0
        signal[hawk_exhaustion] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"