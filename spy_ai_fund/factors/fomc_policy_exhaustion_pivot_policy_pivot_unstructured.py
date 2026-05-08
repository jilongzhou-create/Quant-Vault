import numpy as np
import pandas as pd

class FomcPolicyExhaustionPivotFactor:
    """Fomc Policy Exhaustion Pivot (policy_pivot/unstructured)

    逻辑: 捕捉美联储政策预期的"趋势衰竭与剧烈转向"。对于低频阶梯状的FOMC声明情绪NLP得分，计算其动量变化。如果上一次会议处于鹰派收紧状态(压制股市)，而本次突然发生显著的鸽派跃升(差分大幅翻正)，标志着鹰派彻底衰竭和流动性预期的突然逆转，触发买入脉冲。同理捕捉鸽派衰竭转向。
    数据: fomc_sentiment
    输出: 鹰派衰竭并大幅转鸽输出 +1.0 (看多), 鸽派衰竭并大幅转鹰输出 -1.0 (看空), 常态休眠输出 0.0
    触发条件: FOMC会议日触发极值反转，多空信号延展10个交易日(约两周的情绪消化窗口)，预期 Trigger Rate 5-15%
    """

    def __init__(self, pivot_threshold: float = 0.10, prev_trend_threshold: float = 0.02, pulse_window: int = 10):
        self.name = 'fomc_policy_exhaustion_pivot'
        self.pivot_threshold = pivot_threshold
        self.prev_trend_threshold = prev_trend_threshold
        self.pulse_window = pulse_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理数据缺失情况
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 前向填充获取最新的情绪得分
        fomc = data['fomc_sentiment'].ffill()
        
        # 铁律：只计算边际变化，禁止使用绝对值
        delta = fomc.diff()

        # 识别发生情绪跳变的FOMC会议生效日 (阈值0.001用于滤除浮点数微小误差)
        is_meeting_day = delta.abs() > 0.001

        # 提取并维持历次FOMC会议的情绪变化量(动量记忆)，以便判断前置状态
        valid_delta = delta.where(is_meeting_day, np.nan).ffill()

        # 获取上一次有效会议的情绪变化方向
        prev_valid_delta = valid_delta.shift(1)

        # 抄底反转：上一次会议还在边际变鹰(<-0.02)，本次会议突然发生大幅转鸽(>=0.10)
        dove_pivot = is_meeting_day & (prev_valid_delta <= -self.prev_trend_threshold) & (delta >= self.pivot_threshold)

        # 逃顶反转：上一次会议还在边际变鸽(>0.02)，本次会议突然发生大幅转鹰(<=-0.10)
        hawk_pivot = is_meeting_day & (prev_valid_delta >= self.prev_trend_threshold) & (delta <= -self.pivot_threshold)

        # 将脉冲信号延展特定的交易日窗口 (通常两周内是市场为政策转向定价的黄金时间)
        dove_pulse = dove_pivot.rolling(window=self.pulse_window, min_periods=1).max() == 1
        hawk_pulse = hawk_pivot.rolling(window=self.pulse_window, min_periods=1).max() == 1

        # 初始化休眠全零Series
        signal = pd.Series(0.0, index=data.index)
        
        # 赋值正负脉冲信号
        signal.loc[hawk_pulse] = -1.0
        signal.loc[dove_pulse] = 1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(pivot_threshold={self.pivot_threshold}, pulse_window={self.pulse_window})"