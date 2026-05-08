import numpy as np
import pandas as pd

class FomcSentimentReversalPulseFactor:
    """FOMC情绪突变脉冲因子 (unstructured/NLP Sentiment)

    逻辑: 捕捉美联储FOMC声明情绪的极端超预期反转。FOMC得分为阶梯状低频前填数据，若使用绝对值将导致持续多/空的死板持仓。本因子将逻辑转化为脉冲事件：当短期情绪边际变化创下历史极值（Z-Score > 2.5），并且打破了前期的政策定力（符号反转），且正好在议息日发生跳跃时，输出“狙击手”级别的买卖信号。
    数据: fomc_sentiment
    触发: 5日变化量 Z-Score > 2.5 (极值条件) + 前值符号相反 (衰竭与反转) + 当日出现阶跃 (脉冲瞬发条件)
    输出: +1.0 (鹰转极度鸽派，看多美债), -1.0 (鸽转极度鹰派，看空美债)
    """

    def __init__(self, window=252, diff_window=5, z_thresh=2.5):
        self.name = 'fomc_sentiment_reversal_pulse'
        self.window = window
        self.diff_window = diff_window
        self.z_thresh = z_thresh

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        sentiment = data['fomc_sentiment'].ffill()
        
        # 核心铁律3: 边际变化 (Marginal Change Only)
        # 衡量过去一周的政策预期边际变化总量
        sentiment_diff = sentiment.diff(self.diff_window)
        
        # 核心铁律1: 零值休眠 (Sniper Pulse)
        # NLP得分为T+1落库的阶梯数据，全年约8次变化。只有变化当日 diff(1) != 0。
        # 此条件将强行把信号压缩为极其稀疏的单日脉冲，杜绝了连续持仓的可能。
        daily_jump = sentiment.diff(1)
        
        # 滚动的极值界定
        roll_mean = sentiment_diff.rolling(self.window).mean()
        roll_std = sentiment_diff.rolling(self.window).std()
        
        # 防止分母为0
        z_score = (sentiment_diff - roll_mean) / (roll_std + 1e-6)
        
        # 核心铁律2: 二阶导数/衰竭反转 (Anti-Catch-Falling-Knife)
        # 前一日的预期状态 (判断是否是从旧周期中反转过来，而非趋势延续)
        prev_sentiment = sentiment.shift(1)
        
        # 多头触发: FOMC出现超预期的鸽派跳跃 (Z > 2.5) + 先前市场预期偏鹰 (prev < 0) + 跳跃发生在当日 (daily_jump > 0)
        bull_cond = (z_score > self.z_thresh) & (prev_sentiment < 0.0) & (daily_jump > 0.0)
        
        # 空头触发: FOMC出现超预期的鹰派跳跃 (Z < -2.5) + 先前市场预期偏鸽 (prev > 0) + 跳跃发生在当日 (daily_jump < 0)
        bear_cond = (z_score < -self.z_thresh) & (prev_sentiment > 0.0) & (daily_jump < 0.0)
        
        signal.loc[bull_cond] = 1.0
        signal.loc[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, diff_window={self.diff_window}, z_thresh={self.z_thresh})"