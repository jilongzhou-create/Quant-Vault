import numpy as np
import pandas as pd

class FomcPivotPricingPulseFactor:
    """FomcPivotPricingPulse (policy_pivot/unstructured)

    逻辑: 捕捉美联储FOMC声明文本情绪的边际剧变(鸽派或鹰派突变)。由于文本情绪呈低频阶梯状，当其发生超过经济学阈值的跳跃时，表明政策倾向发生实质性转向，市场通常需要数日来对新的流动性预期进行重新定价。因子在突变发生的瞬间及其后极短窗口内输出顺势脉冲信号。
    数据: fomc_sentiment
    输出: 鸽派突变为+1.0(流动性冲量看多)，鹰派突变为-1.0(紧缩预期看空)
    触发条件: FOMC情绪单日变动 >= 0.2 或 <= -0.2，持续5个交易日，预期 Trigger Rate 约 5% - 10%
    """

    def __init__(self, diff_threshold: float = 0.2, pulse_window: int = 4):
        self.name = 'fomc_pivot_pricing_pulse'
        # 0.2 的变动意味着文本情绪出现显著的措辞修改（如新增降息指引或删减紧缩条件）
        self.diff_threshold = diff_threshold
        # 定价窗口：事件发生当天 + 随后的 4 个交易日（共5天的一周重定价期）
        self.pulse_window = pulse_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 数据缺失保护
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        sentiment = data['fomc_sentiment']
        
        # 【边际变化铁律】: 计算动量变化，捕捉低频阶梯状数据的边际反转
        sentiment_diff = sentiment.diff()
        prev_sentiment = sentiment.shift(1)
        
        # 识别鸽派突变：边际显著变鸽，且前值尚未处于极度鸽派状态(防止市场已充分Price-in)
        is_dovish_pivot = (sentiment_diff >= self.diff_threshold) & (prev_sentiment < 0.5)
        
        # 识别鹰派突变：边际显著变鹰，且前值尚未处于极度鹰派状态
        is_hawkish_pivot = (sentiment_diff <= -self.diff_threshold) & (prev_sentiment > -0.5)
        
        # 初始化极短脉冲信号 (默认休眠 0.0)
        signal_raw = pd.Series(0.0, index=data.index)
        signal_raw.loc[is_dovish_pivot] = 1.0
        signal_raw.loc[is_hawkish_pivot] = -1.0
        
        # 【零值休眠铁律】: 延展极短的市场定价窗口
        # 使用 ffill 限定 limit，将脉冲影响维持极短几天，窗口过后自动衰竭回落至 0.0
        signal_raw = signal_raw.replace(0.0, np.nan)
        signal_pulse = signal_raw.ffill(limit=self.pulse_window).fillna(0.0)
        
        signal_pulse.name = self.name
        return signal_pulse

    def __repr__(self):
        return f"{self.__class__.__name__}(diff_threshold={self.diff_threshold}, pulse_window={self.pulse_window})"