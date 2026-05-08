import numpy as np
import pandas as pd

class FomcPivotSentimentPulseFactor:
    """Fomc Pivot Sentiment Pulse Factor (policy_pivot/unstructured)

    逻辑: 捕捉美联储FOMC声明的鹰鸽情绪突变。低频阶梯期返回0。只有当情绪发生剧烈跳变(Dovish/Hawkish Pivot)时，才触发预期反转的短线操作脉冲。
    数据: [fomc_sentiment]
    输出: 鸽派突变(边际改善)产生+1.0，鹰派突变(边际恶化)产生-1.0
    触发条件: 情绪单次跳变幅度超过0.25(代表政策指引发生实质性修改)，信号维持5个交易日，预期 Trigger Rate 约 8%
    """

    def __init__(self, diff_threshold=0.25, pulse_days=5):
        self.name = 'fomc_pivot_sentiment_pulse'
        self.diff_threshold = diff_threshold
        self.pulse_days = pulse_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理数据缺失情况，默认返回0.0
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 前向填充因为它是非会议日不变的阶梯状数据
        sentiment = data['fomc_sentiment'].ffill()
        
        # 铁律8: 边际变化铁律。对于FOMC阶梯状数据，绝对禁止直接输出绝对值！必须使用 .diff()
        sentiment_margin = sentiment.diff()

        # 触发器：情绪跳变达到具有经济学意义的实质性修改门槛
        dovish_jump = sentiment_margin >= self.diff_threshold
        hawkish_jump = sentiment_margin <= -self.diff_threshold

        # 脉冲维持期：让信号覆盖政策预期被快速消化的极短窗口（1周/5个交易日）
        # 铁律6: 常态休眠，信号在平时完全归零
        dovish_pulse = dovish_jump.rolling(window=self.pulse_days, min_periods=1).max() == 1
        hawkish_pulse = hawkish_jump.rolling(window=self.pulse_days, min_periods=1).max() == 1

        signal = pd.Series(0.0, index=data.index)
        signal.loc[dovish_pulse] = 1.0
        signal.loc[hawkish_pulse] = -1.0
        
        # 防止极端情况下同一窗口期出现重叠冲突
        signal.loc[dovish_pulse & hawkish_pulse] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(diff_threshold={self.diff_threshold}, pulse_days={self.pulse_days})"