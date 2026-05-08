import numpy as np
import pandas as pd

class NlpFomcSentimentDigestedShockFactor:
    """NLP FOMC Sentiment Digested Shock (unstructured/unstructured)

    逻辑: 捕捉美联储FOMC声明文本鹰鸽情绪的极端边缘突变。为避免在FOMC当天的剧烈博弈中接飞刀(如声明鸽派但随后发布会转鹰), 因子必须等待单日边际变化归零(即文本情绪发布后经过1-2天的市场初步消化)且短期累计突变势能仍处于极值时，才触发顺势脉冲，捕捉确定性二阶趋势。
    数据: fomc_sentiment (NLP文本情绪得分)
    触发: fomc_sentiment 3日边际变化量的 Z-Score > 2.5 (极端脉冲) 且 1日边际变化量近乎为 0 (单日抛压/买盘衰竭，短期情绪博弈结束)
    输出: +1.0 看多(极端鸽派突变且已消化), -1.0 看空(极端鹰派突变且已消化), 常态 0.0 (触发率稳定在 ~6%)
    """

    def __init__(self):
        self.name = 'nlp_fomc_sentiment_digested_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 必须处理数据缺失的情况
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)

        fomc = data['fomc_sentiment']

        # Rule 3: 边际变化绝对铁律 (禁止直接使用阶梯状低频数据的绝对值)
        # 计算3日累计突变势能，捕捉低频会议数据发布前后的阶梯跳跃
        chg_3d = fomc.diff(3)
        # 计算单日边际变化，用于严格判断最新日内动能是否已经衰竭
        chg_1d = fomc.diff(1)

        # 计算 252 日(一年)滚动 Z-Score 衡量该次文本情绪突变在宏观周期中的极端性
        roll_mean = chg_3d.rolling(window=252, min_periods=21).mean()
        roll_std = chg_3d.rolling(window=252, min_periods=21).std()
        
        # 加上 1e-8 防止在低频真空期出现除以零的错误
        z_score = (chg_3d - roll_mean) / (roll_std + 1e-8)

        # Rule 2: 二阶导数绝对铁律 (极端极值 + 衰竭确认，绝不接当天的飞刀)
        # 条件1: 突变势能达到统计学极端 (Z > 2.5 鸽派突变 或 Z < -2.5 鹰派突变)
        # 条件2: 最新单日变化已平息 (说明FOMC决议带来的首日情绪跳跃已发生在1-2天前，市场完成了消化Price-in)
        is_extreme_dovish = z_score > 2.5
        is_extreme_hawkish = z_score < -2.5
        is_exhausted = chg_1d.abs() < 1e-5

        # Rule 1: 零值休眠绝对铁律 (常态信号必须为 0.0，仅在极端脉冲事件后输出非零信号)
        signal = pd.Series(0.0, index=data.index)

        # 狙击手级脉冲：双重条件共振时才出击
        signal[is_extreme_dovish & is_exhausted] = 1.0
        signal[is_extreme_hawkish & is_exhausted] = -1.0

        signal.name = self.name
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}()"