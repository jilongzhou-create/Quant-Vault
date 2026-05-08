import numpy as np
import pandas as pd

class FomcTonePivotPulseFactor:
    """FOMC 预期突变脉冲因子 (policy_pivot/unstructured)

    逻辑: 捕捉美联储货币政策口吻的剧烈反转。FOMC声明情绪低频且呈阶梯状, 市场的剧烈重定价只发生在情绪得分发生显著边际变化(Jump)的瞬间。当鹰鸽情绪得分单次变动超过0.15时(代表政策语气的实质性转变), 触发为期3天的顺势脉冲, 捕捉短线流动性预期的急速修正。
    数据: fomc_sentiment (非会议日前向填充的阶梯数据)
    输出: 鸽派突变(边际变鸽)产生+1.0脉冲(看多美股), 鹰派突变(边际变鹰)产生-1.0脉冲(看空美股)
    触发条件: fomc_sentiment.diff()绝对值 > 0.15, 信号持续3个交易日。预期 Trigger Rate 约 6%-10% (每年约4-6次意外变动 x 3天)。
    """

    def __init__(self, jump_threshold: float = 0.15, pulse_window: int = 3):
        """
        :param jump_threshold: 情绪得分变化的突变阈值 (范围是-1到1, 0.15代表15%的显著口吻偏转)
        :param pulse_window: 政策拐点发生后的动量消化期(交易日数)
        """
        self.name = 'fomc_tone_pivot_pulse'
        self.jump_threshold = jump_threshold
        self.pulse_window = pulse_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal

        # 提取 FOMC 情绪得分 (前向填充的阶梯状数据)
        sentiment = data['fomc_sentiment'].ffill()
        
        # 【边际变化铁律】: 绝对禁止使用绝对值, 必须计算动量变化(差分)
        # 因为数据是非会议日ffill的，所以 diff 只在会议T+1生效日当天不为0
        sentiment_diff = sentiment.diff()

        # 识别鸽派突变 (大幅向多头方向修正)
        dovish_jump = sentiment_diff >= self.jump_threshold
        
        # 识别鹰派突变 (大幅向空头方向修正)
        hawkish_jump = sentiment_diff <= -self.jump_threshold

        # 【零值休眠铁律】: 信号必须是狙击手级别的脉冲
        # 使用 rolling max 将跳跃当天的信号向后延续极短的几天 (pulse_window)，让市场消化预期
        # dovish_jump 包含 True/False, 转换为整数 1/0
        dovish_pulse = dovish_jump.astype(int).rolling(window=self.pulse_window, min_periods=1).max() > 0
        hawkish_pulse = hawkish_jump.astype(int).rolling(window=self.pulse_window, min_periods=1).max() > 0

        # 赋值信号输出
        signal[dovish_pulse] = 1.0
        signal[hawkish_pulse] = -1.0

        # 处理可能的数据缺失带来的 NA
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(jump_threshold={self.jump_threshold}, pulse_window={self.pulse_window})"