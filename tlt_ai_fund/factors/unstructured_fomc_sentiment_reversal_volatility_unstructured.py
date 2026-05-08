import numpy as np
import pandas as pd

class UnstructuredFomcSentimentReversalFactor:
    """Unstructured FOMC Sentiment Reversal Factor (NLP Policy Volatility)

    逻辑: 捕捉央行货币政策预期的极端突变。对于低频阶梯状的NLP情绪得分，其边际跃升本身即代表政策波动率的脉冲(Policy Shock)。根据边际变化铁律，利用5日动量(diff)衡量情绪的极端跳跃(Z-Score > 2.5)，配合情绪的极性跨越(如从鹰派<0瞬间翻转为鸽派>0)，以确认旧预期的彻底衰竭与新预期的爆发。利用5日差分自然形成的5天生命周期，完美构成狙击手级别脉冲，将目标Trigger Rate控制在5%-15%区间。
    数据: fomc_sentiment
    触发: 5日情绪变化量 Z-Score > 2.5 且从负转正 -> 鹰转鸽(极度宽松)；Z-Score < -2.5 且从正转负 -> 鸽转鹰(极度紧缩)。
    输出: [-1.0, 1.0] 脉冲信号，正值看多美债(TLT)，负值看空，常态严格为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_sentiment_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始信号全设为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查数据完备性
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # 提取 FOMC 情绪得分并前向填充，确保在非会议日保留最新预期
        fomc = data['fomc_sentiment'].ffill()
        
        # 铁律3: 边际变化。绝对禁止直接使用绝对值，使用 5 日动量差分捕捉预期突变瞬间
        # 使用 5 日窗口可以在跳跃发生后自然维持 5 天的高动量，形成"极短几天内"的脉冲信号
        fomc_chg_5d = fomc.diff(5).fillna(0.0)
        
        # 计算边际变化的 252 日(一年)滚动 Z-Score，min_periods 设为一个自然月
        roll_mean = fomc_chg_5d.rolling(window=252, min_periods=21).mean()
        roll_std = fomc_chg_5d.rolling(window=252, min_periods=21).std()
        
        zscore = (fomc_chg_5d - roll_mean) / (roll_std + 1e-6)
        
        # 铁律2: 二阶导数 (极值 + 衰竭/反转确认)
        # 对于阶跃预期数据，反转确认表现为绝对极性跨越零轴 (Regime Change)
        
        # 鹰转鸽突变：动量极端看多(>2.5σ) 且 当期情绪转为鸽派(>0) 且 5天前情绪为鹰派(<0)
        cond_dove_reversal = (zscore > 2.5) & (fomc > 0.0) & (fomc.shift(5) < 0.0)
        
        # 鸽转鹰突变：动量极端看空(<-2.5σ) 且 当期情绪转为鹰派(<0) 且 5天前情绪为鸽派(>0)
        cond_hawk_reversal = (zscore < -2.5) & (fomc < 0.0) & (fomc.shift(5) > 0.0)
        
        # 输出脉冲信号
        signal[cond_dove_reversal] = 1.0
        signal[cond_hawk_reversal] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"