import numpy as np
import pandas as pd

class UnstructuredFomcNlpShockConfirmationFactor:
    """FOMC非结构化情绪突变落地脉冲 (unstructured/unstructured)

    逻辑: 捕捉美联储FOMC文本声明情绪极值跳跃(Policy Pivot)引发的重定价时刻。直接使用绝对值会导致持续信号，因此提取NLP鸽派/鹰派情绪的5日边际突变量计算历史Z-Score。为避免在预期最剧烈混乱的当天接飞刀，必须加入二阶衰竭条件：当整体趋势在极值区间，但单日变化幅度已经萎缩并低于近3日均值时，确认政策震撼已落地稳固，此时狙击手精准开火生成+1.0/-1.0脉冲信号。
    数据: fomc_sentiment
    触发: 5日变化量的 252日 Z-Score > 2.5(鸽) 或 < -2.5(鹰) + 单日变化的绝对值 < 其3日移动均值 (二阶动能衰竭)。
    输出: +1.0 (鸽派转向落地，看多美债) / -1.0 (鹰派转向落地，看空美债)，常态为 0.0。
    """

    def __init__(self, diff_window: int = 5, zscore_window: int = 252, z_threshold: float = 2.5):
        self.name = 'unstructured_fomc_nlp_shock_confirmation'
        self.diff_window = diff_window
        self.zscore_window = zscore_window
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # T+1 生效数据，前向填充以应对非会议日的连续性
        sentiment = data['fomc_sentiment'].ffill()
        
        # 铁律3: 边际变化 Only
        # 提取低频阶梯状数据的动量变化，绝对禁止使用其原本的绝对值
        sent_diff = sentiment.diff(self.diff_window)
        
        # 计算动量变化的统计极值水位
        roll_mean = sent_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).mean()
        roll_std = sent_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).std()
        
        # 增加极小常量避免零除问题
        zscore = (sent_diff - roll_mean) / (roll_std + 1e-8)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 单日边际动能衰竭：当前单日变化幅度小于近期移动平均幅度。
        # 对于阶梯状事件数据，这能完美且只提取跳变发生后、预期消化不再发酵的“稳固日”，避开飞刀。
        daily_diff_abs = sentiment.diff(1).abs()
        exhausted = daily_diff_abs < daily_diff_abs.rolling(3).mean()
        
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 只在多空极端情绪跃迁并确认降温的极其稀疏瞬间才触发正负 1.0
        long_pulse = (zscore > self.z_threshold) & exhausted
        short_pulse = (zscore < -self.z_threshold) & exhausted
        
        signal.loc[long_pulse] = 1.0
        signal.loc[short_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(diff_window={self.diff_window}, zscore_window={self.zscore_window}, z_threshold={self.z_threshold})"