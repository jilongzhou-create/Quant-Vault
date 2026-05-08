import numpy as np
import pandas as pd

class FomcSentimentStructuralPivotPulseFactor:
    """FomcSentimentStructuralPivotPulseFactor (policy_pivot/unstructured)

    逻辑: 捕捉美联储货币政策情绪相对于过去一年基准的结构性突变(跨越1年期均线或大幅跳升)。FOMC决议低频且具有突发性,本因子通过识别声明情绪得分在单一交易日内的阶梯式跳跃(边际变化),且该变化突破历史基准,从而在市场定价剧变的最初数天输出脉冲信号。
    数据: [fomc_sentiment]
    输出: 1.0(鸽派结构性转向/加速看多), -1.0(鹰派结构性转向/加速看空)
    触发条件: 情绪得分发生跳跃且相对于1年基线反转或单次跳跃>0.15, 信号维持3天, 预期Trigger Rate约5%-10%
    """

    def __init__(self, pulse_hold_days=3, jump_threshold=0.15, baseline_window=252):
        self.name = 'fomc_sentiment_structural_pivot_pulse'
        self.pulse_hold_days = pulse_hold_days
        self.jump_threshold = jump_threshold
        self.baseline_window = baseline_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'fomc_sentiment' not in data.columns or data['fomc_sentiment'].isna().all():
            return pd.Series(0.0, index=data.index, name=self.name)

        # FOMC情绪分数是低频阶梯状数据, T+1生效, 使用前向填充以确保基线连续性
        sentiment = data['fomc_sentiment'].ffill()
        
        # 计算动量变化(边际跳跃) - 严格遵循只在预期发生改变的瞬间抓取变化的铁律
        sentiment_jump = sentiment.diff()
        
        # 计算1年期(约252个交易日)的情绪基准线, 代表中长期的政策基调历史预期
        baseline_1y = sentiment.rolling(window=self.baseline_window, min_periods=60).mean()
        
        signal = pd.Series(0.0, index=data.index)
        
        prev_sentiment = sentiment.shift(1)
        prev_baseline = baseline_1y.shift(1)
        
        # 仅在阶梯数据发生变化的瞬间(跳跃日)进行判断，严禁在平时输出连续的绝对值信号
        is_jump_day = sentiment_jump.abs() > 1e-4
        
        # 鸽派脉冲 (+1.0):
        # A: 情绪得分发生了极强烈的正向突变 (鸽派突变)
        # B: 情绪得分转向上穿1年期基准线, 且当天发生了实质性正向跳跃 (结构性鸽派转向)
        dovish_jump = sentiment_jump > self.jump_threshold
        dovish_cross = (prev_sentiment < prev_baseline) & (sentiment > baseline_1y) & (sentiment_jump > 0.05)
        dovish_trigger = is_jump_day & (dovish_jump | dovish_cross)
        
        # 鹰派脉冲 (-1.0):
        # A: 情绪得分发生了极强烈的负向突变 (鹰派突变)
        # B: 情绪得分转向下穿1年期基准线, 且当天发生了实质性负向跳跃 (结构性鹰派转向)
        hawkish_jump = sentiment_jump < -self.jump_threshold
        hawkish_cross = (prev_sentiment > prev_baseline) & (sentiment < baseline_1y) & (sentiment_jump < -0.05)
        hawkish_trigger = is_jump_day & (hawkish_jump | hawkish_cross)
        
        # 注入瞬时脉冲信号 (狙击手级)
        signal.loc[dovish_trigger] = 1.0
        signal.loc[hawkish_trigger] = -1.0
        
        # 将突发脉冲向后延展数天 
        # (因为市场往往需要2-3天时间完全消化美联储政策转折声明的细节影响，同时保证Trigger Rate控制在5%-15%)
        signal = signal.replace(0.0, np.nan).ffill(limit=self.pulse_hold_days - 1).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(pulse_hold_days={self.pulse_hold_days}, jump_threshold={self.jump_threshold})"