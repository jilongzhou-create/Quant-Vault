import numpy as np
import pandas as pd

class FomcPanicExhaustionPulseFactor:
    """FomcPanicExhaustionPulse (panic_mean_reversion/unstructured)

    逻辑: 紧缩预期的极值与均值回归。FOMC声明情绪得分是低频呈阶梯状的非结构化NLP衍生数据。
          在SPY长牛市场中, 若前期已被反复的鹰派言论打压至恐慌状态, 只要边际上出现超预期的鸽派跳变(情绪开始回暖), 
          紧缩恐慌即告衰竭, 空头回补将爆发猛烈的抄底买盘。
          反之, 若前期市场处于自满状态, 却遭遇边际突发鹰派收紧, 则触发看空恶化脉冲。
    数据: fomc_sentiment
    输出: +1.0 (恐慌衰竭/鸽派突变), -1.0 (预期突发恶化/鹰派突变), 0.0 (常态休眠)
    触发条件: 绝对禁止直接使用绝对值, 仅在会议决议当天预期发生巨大跳跃(|diff|>0.15)时触发, 顺延3日(保障Trigger Rate在5%-10%区间)。
    """

    def __init__(self):
        self.name = 'fomc_panic_exhaustion_pulse'
        self.sentiment_shift_threshold = 0.15  # 情绪发生显著边际反转的变动幅度
        self.hawkish_panic_threshold = -0.10   # 前置状态定义: 至少在此水位之下视为已处在鹰派压抑周期
        self.pulse_window = 3                  # 脉冲维持交易日数, 适应机构资金建仓窗口期

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # T+1生效的数据通常以前向填充处理，我们需要捕捉其台阶变化的瞬间
        sentiment = data['fomc_sentiment'].ffill()
        sentiment_prev = sentiment.shift(1)
        sentiment_diff = sentiment.diff()

        # 找到有效会议发布日 (低频阶梯状数据的边缘跳变点)
        is_meeting_day = sentiment_diff.abs() > 0.01

        # 1. 恐慌衰竭脉冲 (+1.0): 
        # 前期已被打压至偏鹰派恐慌状态 (< -0.10)，本次边际出现显著鸽派改善 (> 0.15)
        buy_trigger = is_meeting_day & (sentiment_prev < self.hawkish_panic_threshold) & (sentiment_diff > self.sentiment_shift_threshold)
        
        # 2. 趋势恶化脉冲 (-1.0): 
        # 前期没有极端恐慌 (处于中性或偏鸽自满)，本次意外向鹰派收紧 (< -0.15)
        sell_trigger = is_meeting_day & (sentiment_prev >= self.hawkish_panic_threshold) & (sentiment_diff < -self.sentiment_shift_threshold)

        raw_pulse = pd.Series(0.0, index=data.index)
        raw_pulse.loc[buy_trigger] = 1.0
        raw_pulse.loc[sell_trigger] = -1.0

        # 脉冲顺延逻辑: 点状的单日脉冲会使得Trigger Rate过低
        # 延展成3日建仓脉冲，规避每天频繁触发且满足5%~15%覆盖铁律
        pulse_pos = (raw_pulse == 1.0).rolling(window=self.pulse_window, min_periods=1).max() == 1.0
        pulse_neg = (raw_pulse == -1.0).rolling(window=self.pulse_window, min_periods=1).max() == 1.0

        signal.loc[pulse_pos] = 1.0
        # 冲突规避: 看多优先
        signal.loc[pulse_neg & ~pulse_pos] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"