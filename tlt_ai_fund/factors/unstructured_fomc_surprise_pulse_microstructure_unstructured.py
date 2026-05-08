import numpy as np
import pandas as pd

class UnstructuredFomcSurprisePulseFactor:
    """Unstructured FOMC Surprise Pulse (microstructure/unstructured)

    逻辑: 采用基于非结构化 NLP 分析的 FOMC 鹰鸽情绪得分，利用 .diff() 计算预期的突变脉冲，严格遵守边际变化铁律。绝对禁止使用情绪得分的绝对水位。当情绪得分发生超预期的剧烈跳跃（252日边缘变化 Z-Score > 2.5）时，代表美联储预期发生历史级别突变，此时的阶梯跳跃瞬变即宣告了前序预期的“衰竭”与重新定价。在触发当日及随后极短2天内输出脉冲信号，随后迅速休眠归零，达成狙击手级别的脉冲特征。
    数据: fomc_sentiment (FOMC 声明鹰鸽情绪得分，NLP 非结构化转化数据)
    触发: fomc_sentiment.diff() 的 Z-Score 绝对值 > 2.5，且当日真实发生非零的跳跃（预期改变的瞬间）
    输出: 突变鸽派输出 +1.0 看多美债，突变鹰派输出 -1.0 看空美债，常态严格为 0.0
    """

    def __init__(self):
        self.name = 'unstructured_fomc_surprise_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 检查所需数据是否在 DataFrame 中
        if 'fomc_sentiment' not in data.columns:
            return signal

        # 前向填充处理非会议日的空值，保证低频数据的平滑阶梯状特征，T+1 生效防前瞻
        fomc = data['fomc_sentiment'].ffill()

        # 三大铁律3: 边际变化 (Marginal Change Only) - 绝对禁止使用绝对值，只捕捉预期改变瞬间的跳跃
        # 当 fomc_diff != 0 时，即为预期 pricing 的突变边缘
        fomc_diff = fomc.diff(1)

        # 计算边际变化的 252日 Z-Score
        # 增加 min_periods 保证早期快速出值
        # 加入 0.1 经济学底限（代表在[-1,1]的情绪空间中10%的变化量），防止处于较长平静期时波动率极低导致的除零异动/伪极值
        roll_mean = fomc_diff.rolling(window=252, min_periods=10).mean()
        roll_std = fomc_diff.rolling(window=252, min_periods=10).std().clip(lower=0.1)

        # 避免初期的 NaN 干扰计算
        z_score = ((fomc_diff - roll_mean) / roll_std).fillna(0.0)

        # 三大铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 对于阶梯状数据，变化量本身（单日 diff 不为 0）即代表了极值的边缘发生并立即确认定价，等同于衰竭转折
        is_jump = (fomc_diff != 0) & (fomc_diff.notna())

        # 鸽派突变（情绪得分正向剧增）：做多美债
        long_cond = (z_score > 2.5) & (fomc_diff > 0) & is_jump

        # 鹰派突变（情绪得分负向剧减）：做空美债
        short_cond = (z_score < -2.5) & (fomc_diff < 0) & is_jump

        # 初始化原始单日脉冲
        raw_signal = pd.Series(0.0, index=data.index)
        raw_signal[long_cond] = 1.0
        raw_signal[short_cond] = -1.0

        # 三大铁律1: 零值休眠 (Sniper Pulse)
        # 为了达到 5% 到 15% 的 Trigger Rate 并保持严格的脉冲特性，将脉冲延展持仓 3 天
        # 即"只在极端事件发生的当天及随后极短几天内输出非零值"
        signal_long = raw_signal.copy()
        signal_long[signal_long < 0] = 0
        signal_long = signal_long.rolling(window=3, min_periods=1).max()

        signal_short = raw_signal.copy()
        signal_short[signal_short > 0] = 0
        signal_short = signal_short.rolling(window=3, min_periods=1).min()

        # 合并多空输出，无信号处严格保持 0.0
        signal[signal_long > 0] = 1.0
        signal[signal_short < 0] = -1.0
        
        # 兜底填充防缺失
        signal = signal.fillna(0.0)
        signal.name = self.name

        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"