import numpy as np
import pandas as pd

class UnstructuredFomcSentimentMicroShockFactor:
    """非结构化FOMC情绪微观反转脉冲因子 (microstructure/unstructured)

    逻辑: 将央行FOMC声明的非结构化文本情绪(NLP Sentiment)突变转化为微观流动性冲击的脉冲信号。
          使用 diff(5) 自然将低频的单日预期跳跃展宽为连续5天的微观交易吸收周期，保证Trigger Rate适中。
          只有当情绪动量发生极端突变(Z-Score>2.5)且情绪绝对水位完成从负向正(鹰转鸽)的反转穿越时，
          才标志着微观结构上空头抛压的彻底衰竭和流动性预期的重定价，从而触发脉冲买入信号。
    数据: fomc_sentiment (非结构化文本情绪得分)
    触发: 5日变化量(动量)的 252日 Z-Score > 2.5 且 情绪由负(鹰)转正(鸽) → 看多脉冲 +1.0
          5日变化量(动量)的 252日 Z-Score < -2.5 且 情绪由正(鸽)转负(鹰) → 看空脉冲 -1.0
    输出: [-1.0, 1.0] 的极短期脉冲，常态下严格保持 0.0 (零值休眠)。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_sentiment_micro_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号强制全为 0.0 (铁律1: 零值休眠，狙击手脉冲常态保持静默)
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # 前向填充填补非会议日的空缺
        fomc = data['fomc_sentiment'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝对禁止直接输出阶梯绝对值，必须使用差分。
        # 这里使用 5日差分 捕捉跳跃，它能将会议当天的跃变自然展宽为随后的 5天吸收脉冲
        delta_5 = fomc.diff(5)
        
        # 计算 252个交易日(约一年)的滚动 Z-Score 来衡量边际突变的极端性 (防前瞻偏差)
        roll_mean = delta_5.rolling(window=252, min_periods=21).mean()
        roll_std = delta_5.rolling(window=252, min_periods=21).std().replace(0.0, np.nan)
        zscore = (delta_5 - roll_mean) / roll_std
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 多头信号触发: 
        # 条件1 (极值): 边际变化量出现历史罕见的向上飙升 (鸽派突变)
        cond_bull_shock = zscore > 2.5
        # 条件2 (衰竭/反转确认): 绝对情绪水位在同一周期内完成从鹰派(负)向鸽派(正)的实质性穿越
        cond_bull_reversal = (fomc > 0.0) & (fomc.shift(5) < 0.0)
        
        # 空头信号触发:
        # 条件1 (极值): 边际变化量出现历史罕见的向下暴跌 (鹰派突变)
        cond_bear_shock = zscore < -2.5
        # 条件2 (衰竭/反转确认): 绝对情绪水位在同一周期内完成从鸽派(正)向鹰派(负)的实质性穿越
        cond_bear_reversal = (fomc < 0.0) & (fomc.shift(5) > 0.0)
        
        # 生成非零脉冲信号，不满足条件区域自然维持上方的初始化 0.0
        signal.loc[cond_bull_shock & cond_bull_reversal] = 1.0
        signal.loc[cond_bear_shock & cond_bear_reversal] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"