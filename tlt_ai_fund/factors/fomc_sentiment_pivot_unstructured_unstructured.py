import numpy as np
import pandas as pd

class FomcSentimentPivotFactor:
    """FOMC情绪突变因子 (unstructured/nlp_sentiment)

    逻辑: 捕捉美联储货币政策预期的极端跳跃。FOMC情绪得分是低频阶梯状数据，绝对值具有高度滞后性且会被市场提前Price-in。只有在预期发生边际突变的瞬间（变化量极端爆发）且原有的政策立场发生反转/衰竭（之前处于对立面）时，才触发脉冲信号，从而狙击极端的定价差。
    数据: fomc_sentiment
    触发: 5日边际变化量 Z-Score 的绝对值 > 2.5 (极值条件) + 触发日发生实际阶梯跳跃且过往5天立场与当前突变方向相反 (衰竭反转条件)
    输出: 脉冲型，鸽派反转突变看多TLT输出+1.0，鹰派反转突变看空TLT输出-1.0。脉冲持续5天以满足5%~15%的触发率，其余时间严格零值休眠 (0.0)
    """

    def __init__(self):
        self.name = 'fomc_sentiment_pivot_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号必须为全 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # 前向填充缺失值以保持阶梯状结构，防前瞻且确保可比较
        fomc = data['fomc_sentiment'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝对禁止使用情绪得分的绝对水位，必须使用 5 日变化量来捕捉预期动量突变
        fomc_diff_5d = fomc.diff(5)
        
        # 计算边际变化的极端程度 (252日滚动 Z-Score)
        roll_mean = fomc_diff_5d.rolling(window=252, min_periods=21).mean()
        roll_std = fomc_diff_5d.rolling(window=252, min_periods=21).std()
        zscore_5d = (fomc_diff_5d - roll_mean) / (roll_std + 1e-6)
        
        # 识别实质性的预期改变瞬间 (仅在阶梯跳跃发生的瞬间触发，过滤 diff 窗口遗留的连续值)
        fomc_step = fomc.diff(1)
        is_jump_day = fomc_step.abs() > 0.05
        
        # 铁律2: 二阶导数 (极值 + 衰竭/反转)
        # 鸽派突变脉冲 (+1.0)：向鸽派的极端跳跃 (Z-Score > 2.5)，且前期情绪处于鹰派阵营 (衰竭反转)
        dovish_pivot = (
            is_jump_day & 
            (zscore_5d > 2.5) & 
            (fomc.shift(5) < 0)
        )
        
        # 鹰派突变脉冲 (-1.0)：向鹰派的极端跳跃 (Z-Score < -2.5)，且前期情绪处于鸽派阵营 (衰竭反转)
        hawkish_pivot = (
            is_jump_day & 
            (zscore_5d < -2.5) & 
            (fomc.shift(5) > 0)
        )
        
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 仅在反转跳跃日生成瞬间买卖脉冲
        raw_signal = pd.Series(0.0, index=data.index)
        raw_signal[dovish_pivot] = 1.0
        raw_signal[hawkish_pivot] = -1.0
        
        # 将狙击脉冲向后延展4天 (单次突变总计影响5天)，涵盖市场预期消化期
        # 并以此将每年低频的触发次数提升至目标 5% ~ 15% 的 Trigger Rate 范围内
        active_signal = raw_signal.replace(0.0, np.nan).ffill(limit=4).fillna(0.0)
        
        signal = active_signal
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"