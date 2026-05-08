import numpy as np
import pandas as pd

class FomcSentimentTrendShiftPulseFactor:
    """Fomc Sentiment Trend Shift Pulse (policy_pivot/unstructured)

    逻辑: 捕捉美联储政策情绪在经历了前期明显偏向(鹰/鸽)后的首次反向显著跳跃。当过去3个月处于鹰派压抑期且边际突变转鸽时, 市场流动性预期重估, 产生看多脉冲; 处于鸽派乐观期且边际突变转鹰时看空。
    数据: fomc_sentiment
    输出: 边际政策预期反转瞬间的脉冲信号 [-1.0, 1.0]
    触发条件: FOMC会议情绪得分日环比跳跃 > 0.1(转鸽)或 < -0.1(转鹰), 且前期存在极端情绪水位。脉冲持续 5 天, 预期 Trigger Rate 控制在 5%-15%
    """

    def __init__(self, jump_threshold=0.1, regime_window=63, hawkish_threshold=-0.2, dovish_threshold=0.2, hold_days=5):
        self.name = 'fomc_sentiment_trend_shift_pulse_policy_pivot_unstructured'
        # jump_threshold: 情绪边际跳跃的幅度阈值
        self.jump_threshold = jump_threshold
        # regime_window: 评估前期情绪环境的时间窗口(约3个月)
        self.regime_window = regime_window
        # hawkish_threshold/dovish_threshold: 认定前期存在极端情绪的门槛
        self.hawkish_threshold = hawkish_threshold
        self.dovish_threshold = dovish_threshold
        # hold_days: 市场消化突变信息的脉冲持续期
        self.hold_days = hold_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 字段检查
        if 'fomc_sentiment' not in data.columns:
            return signal

        # 处理阶梯状数据
        sentiment = data['fomc_sentiment'].ffill()
        
        # 边际变化铁律: 只关注突变的动量变化
        delta = sentiment.diff()

        # 计算过去窗口期的极端情绪水位 (shift(1) 不含当日, 避免前向偏倚和当日跳跃的影响)
        past_min = sentiment.shift(1).rolling(window=self.regime_window, min_periods=10).min()
        past_max = sentiment.shift(1).rolling(window=self.regime_window, min_periods=10).max()

        # 看多触发: 边际显著跳跃转鸽, 且验证前期处于较鹰派的压抑环境(衰竭后转向)
        bull_trigger = (delta > self.jump_threshold) & (past_min < self.hawkish_threshold)
        
        # 看空触发: 边际显著跳跃转鹰, 且验证前期处于较鸽派的乐观环境(宽松高潮后逆转)
        bear_trigger = (delta < -self.jump_threshold) & (past_max > self.dovish_threshold)

        # 构建瞬时极端脉冲 (零值休眠铁律)
        raw_signal = pd.Series(0.0, index=data.index)
        raw_signal.loc[bull_trigger] = 1.0
        raw_signal.loc[bear_trigger] = -1.0

        # 延展持仓期，使得信号在预期重估的极短窗口内保持脉冲
        # 使用 replace+ffill 保证严格延展 N-1 天, 其余时间恢复 0.0
        signal = raw_signal.replace(0.0, np.nan).ffill(limit=self.hold_days - 1).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(jump_threshold={self.jump_threshold}, hold_days={self.hold_days})"