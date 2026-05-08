import numpy as np
import pandas as pd

class FomcSentimentBreakoutPulseFactor:
    """FOMC情绪突破脉冲因子 (policy_pivot/unstructured)

    逻辑: 捕捉FOMC声明的鹰鸽情绪不仅发生边际跳跃，且一举突破过去半年均值的"破局点"。这标志着美联储政策预期发生中期级别（而不仅是短期）的实质性转向，市场流动性预期将被重估。
    数据: fomc_sentiment
    输出: +1.0 看多美股(突破性鸽派表态), -1.0 看空美股(突破性鹰派表态)
    触发条件: FOMC情绪跳跃幅度>0.15且绝对水平突破半年均值至少0.1，触发后维持5天，预期Trigger Rate控制在5%-10%
    """

    def __init__(self, diff_threshold=0.15, breakout_margin=0.1, ma_window=126, hold_days=5):
        self.name = 'fomc_sentiment_breakout_pulse'
        self.diff_threshold = diff_threshold
        self.breakout_margin = breakout_margin
        self.ma_window = ma_window
        self.hold_days = hold_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        sentiment = data['fomc_sentiment'].ffill()
        
        # 铁律: 必须基于阶梯状数据的边际变化触发信号
        sentiment_diff = sentiment.diff()
        
        # 识别会议导致的情绪变化瞬间 (跳跃)
        is_jump = sentiment_diff.abs() > 1e-4
        
        # 计算过去半年(126个交易日)的情绪中期预期均值，使用shift(1)避免使用包含当天的信息
        mean_sentiment = sentiment.rolling(window=self.ma_window, min_periods=21).mean().shift(1)
        
        # 多头触发：边际大幅变鸽 (动量剧变) + 当前情绪一举突破中期预期 (破局)
        bull_trigger = is_jump & (sentiment_diff > self.diff_threshold) & (sentiment > mean_sentiment + self.breakout_margin)
        
        # 空头触发：边际大幅变鹰 (动量剧变) + 当前情绪一举跌破中期预期 (破局)
        bear_trigger = is_jump & (sentiment_diff < -self.diff_threshold) & (sentiment < mean_sentiment - self.breakout_margin)
        
        # 初始化零值信号
        signal = pd.Series(0.0, index=data.index)
        
        # 写入极值脉冲
        signal.loc[bull_trigger] = 1.0
        signal.loc[bear_trigger] = -1.0
        
        # 狙击手脉冲：维持极短时间(5天)以匹配流动性发酵窗口，其他时间必须休眠归零
        signal = signal.replace(0.0, np.nan).ffill(limit=self.hold_days - 1).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(diff_threshold={self.diff_threshold}, breakout_margin={self.breakout_margin}, ma_window={self.ma_window})"