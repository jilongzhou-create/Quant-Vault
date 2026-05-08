import numpy as np
import pandas as pd

class UnstructuredFomcSentimentReversalPulseFactor:
    """FOMC情绪极性反转脉冲因子 (unstructured/unstructured)

    逻辑: 捕捉美联储货币政策态度的根本性反转。FOMC声明情绪分是典型的低频阶梯数据，因此严格拒绝绝对值，
          仅通过单日边际突变 (diff) 的极端 Z-Score 捕捉 Shock 瞬间。为防止接飞刀，强制要求情绪必须
          发生“极性反转”(跨越零轴，即从鹰派<0突然反转为鸽派>0)，代表旧政策周期的彻底衰竭与新周期的确认。
    数据: fomc_sentiment (FOMC声明鹰鸽情绪得分, [-1, 1], 1=鸽, -1=鹰)
    触发: sentiment.diff(1) 的 252日 Z-Score > 2.5 (边缘剧变) 且 前值<0 且 当前值>0 (状态反转/衰竭)
    输出: +1.0 (鹰转鸽，看多美债) / -1.0 (鸽转鹰，看空美债) / 0.0 (常态休眠，纯正脉冲)
    """

    def __init__(self):
        self.name = 'unstructured_fomc_sentiment_reversal_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化纯零 Series，遵守零值休眠铁律
        signal = pd.Series(0.0, index=data.index)

        if 'fomc_sentiment' not in data.columns:
            return signal

        # 情绪得分为离散跳跃数据，向前填充保持状态
        sentiment = data['fomc_sentiment'].ffill()

        # 铁律3: 边际变化 (绝对禁止使用绝对值作为信号)
        # 使用单日变化量来捕捉 FOMC 日的瞬间跳跃
        chg_1d = sentiment.diff(1)

        # 计算边际变化的历史 Z-Score (使用 min_periods=60 保证初期有数据)
        # 由于非会议日 chg_1d 为 0，均值趋近于0，标准差很小，会议日的跳跃会产生极大的 Z-Score
        roll_mean = chg_1d.rolling(window=252, min_periods=60).mean()
        roll_std = chg_1d.rolling(window=252, min_periods=60).std()
        
        # 加上 1e-6 防止除以零
        z_score = (chg_1d - roll_mean) / (roll_std + 1e-6)

        # 记录反转前的状态
        prev_sentiment = sentiment.shift(1)

        # 铁律2: 二阶导数 (极值突变 + 衰竭/反转确认)
        # 看多条件 (美债上涨): 发生极端的边际鸽派突变 (Z > 2.5) + 旧有的鹰派预期衰竭并反转 (前值 < 0 且 当前值 > 0)
        long_cond = (z_score > 2.5) & (prev_sentiment < 0.0) & (sentiment > 0.0)

        # 看空条件 (美债下跌): 发生极端的边际鹰派突变 (Z < -2.5) + 旧有的鸽派预期衰竭并反转 (前值 > 0 且 当前值 < 0)
        short_cond = (z_score < -2.5) & (prev_sentiment > 0.0) & (sentiment < 0.0)

        # 铁律1: 狙击手脉冲，仅在触发当日赋值
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"