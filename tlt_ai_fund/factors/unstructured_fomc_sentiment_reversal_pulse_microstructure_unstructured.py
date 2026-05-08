import numpy as np
import pandas as pd

class UnstructuredFomcSentimentReversalPulseFactor:
    """非结构化FOMC情绪反转脉冲因子 (microstructure/unstructured)

    逻辑: 采用基于大语言模型(LLM)解析的央行 FOMC 声明情绪得分 (fomc_sentiment)。遵循三大铁律，低频阶梯状数据绝对禁止使用绝对水位，市场仅对预期边际突变定价。当 FOMC 情绪得分的 5 日变化量出现极端偏离 (Z-Score > 2.5)，且情绪发生跨零轴的实质性反转时，释放极短期的交易脉冲，捕获宏观情绪重定价的狙击窗口。
    数据: fomc_sentiment (1.0=极度鸽派看多美债, -1.0=极度鹰派看空美债)
    触发: 
      看多(+1.0): 5日突变Z-Score > 2.5 且 情绪从鹰派(负)转鸽派(正)
      看空(-1.0): 5日突变Z-Score < -2.5 且 情绪从鸽派(正)转鹰派(负)
    输出: [-1.0, 1.0] 的极短期脉冲信号，通过 5 日差分自然产生时长 5 天的交易窗口，满足 5%-15% 的 Trigger Rate 目标。
    """

    def __init__(self, diff_window=5, zscore_window=252, zscore_threshold=2.5):
        self.name = 'unstructured_fomc_sentiment_reversal_pulse'
        self.diff_window = diff_window
        self.zscore_window = zscore_window
        self.zscore_threshold = zscore_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # 填充非会议日的缺失值，形成阶梯状数据
        fomc = data['fomc_sentiment'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change)
        # 计算 5 日变化量捕捉预期跳跃，5日窗口不仅能平滑噪声，还能自动将跳跃延续为极短期脉冲
        fomc_diff = fomc.diff(self.diff_window)
        
        # 计算动量变化的 252 日 Z-Score，衡量此次会议超预期的程度
        diff_mean = fomc_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).mean()
        diff_std = fomc_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).std()
        
        # 加上极小值避免除以0
        diff_zscore = (fomc_diff - diff_mean) / (diff_std + 1e-8)
        
        # 获取突变前的情绪基准与当前情绪
        prev_fomc = fomc.shift(self.diff_window)
        curr_fomc = fomc
        
        # 铁律2: 二阶导数与反转衰竭条件 (明确跨零轴反转)
        # 看多脉冲 (+1.0): 情绪从偏鹰(<0)大幅突变并反转为偏鸽(>0)
        buy_cond = (diff_zscore > self.zscore_threshold) & (prev_fomc < 0) & (curr_fomc > 0)
        
        # 看空脉冲 (-1.0): 情绪从偏鸽(>0)大幅突变并反转为偏鹰(<0)
        sell_cond = (diff_zscore < -self.zscore_threshold) & (prev_fomc > 0) & (curr_fomc < 0)
        
        signal.loc[buy_cond] = 1.0
        signal.loc[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(diff_window={self.diff_window}, zscore_threshold={self.zscore_threshold})"