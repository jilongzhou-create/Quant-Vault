import numpy as np
import pandas as pd

class FomcSentimentSurpriseFactor:
    """FOMC情绪突变脉冲因子 (policy_pivot/unstructured)

    逻辑: 捕捉美联储FOMC声明预期发生剧变的极短窗口。使用NLP分析FOMC声明的鸽鹰情绪(fomc_sentiment)，当单次会议情绪边际变化(.diff())大于指定阈值时，视为政策预期发生鸽派或鹰派反转。该冲击会被市场在随后一周内消化，因此信号仅维持5个交易日，其他时间坚决空仓。
    数据: fomc_sentiment
    输出: 脉冲信号 [-1.0, 1.0]。正值代表鸽派突变看多美股，负值代表鹰派突变看空美股。
    触发条件: fomc_sentiment.diff()绝对值 >= 0.3，触发后维持5个交易日，常态下全为0。预期Trigger Rate约5%-10%。
    """

    def __init__(self, diff_threshold=0.3, hold_days=5):
        self.name = 'fomc_sentiment_surprise'
        # 阈值经济学含义：0.3 代表在 [-1.0, 1.0] 的情绪标尺上发生了15%的剧烈预期修正
        self.diff_threshold = diff_threshold
        # 维持5个交易日，即FOMC剧变后市场消化流动性预期的标准物理时间
        self.hold_days = hold_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理缺失列，直接返回全0序列
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        # 前向填充FOMC低频阶梯状数据
        fomc = data['fomc_sentiment'].ffill()
        
        # 【边际变化铁律】: 绝对禁止看绝对值，只捕捉预期改变的瞬间！
        fomc_diff = fomc.diff()
        
        # 识别短期的鸽派剧变和鹰派剧变
        dovish_jump = (fomc_diff >= self.diff_threshold).astype(float)
        hawkish_jump = (fomc_diff <= -self.diff_threshold).astype(float)
        
        # 【零值休眠铁律】: 脉冲延展 - 将剧变当天的瞬间脉冲向后保持 hold_days 天，其余时间强制回落为0
        dovish_pulse = dovish_jump.rolling(window=self.hold_days, min_periods=1).max()
        hawkish_pulse = hawkish_jump.rolling(window=self.hold_days, min_periods=1).max()
        
        # 合并多空信号 (互相排斥)
        signal = dovish_pulse - hawkish_pulse
        
        # 规范化处理
        signal = signal.clip(-1.0, 1.0).fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(diff_threshold={self.diff_threshold}, hold_days={self.hold_days})"