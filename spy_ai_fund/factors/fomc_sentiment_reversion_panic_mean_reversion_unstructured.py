import numpy as np
import pandas as pd

class FomcSentimentReversionFactor:
    """FOMC情绪极值衰竭突变 (panic_mean_reversion/unstructured)

    逻辑: 极度鹰派(政策恐慌)会持续压制美股，绝对禁止在极端看空期接飞刀。必须等待FOMC情绪在极端鹰派区间首次出现边际改善(.diff() > 0)时(即政策恐慌见顶衰竭瞬间)，才触发极短线的抄底做多脉冲；反之，如果处于极端鸽派且边际转鹰，则触发趋势恶化看空脉冲。
    数据: [fomc_sentiment]
    输出: 1.0 (鹰派极值衰竭/强鸽突变), -1.0 (鸽派极值破灭/强鹰突变), 0.0 (常态休眠)
    触发条件: FOMC预期跳变当天及随后3天输出非零值，由于会议频次较低，信号适当延展以确保目标 Trigger Rate 稳定在 5% 到 10% 左右。
    """

    def __init__(self):
        self.name = 'fomc_sentiment_reversion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 数据缺失校验
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        # 获取阶梯低频的 FOMC NLP 情绪得分，范围 [-1.0, 1.0] (1.0为极度鸽派, -1.0为极度鹰派)
        sent = data['fomc_sentiment']
        
        # 边际变化铁律: 低频前向填充数据绝对禁止直接输出绝对值，必须计算差分捕捉预期改变的瞬间
        sent_diff = sent.diff()
        prev_sent = sent.shift(1)
        
        # 1. 恐慌衰竭抄底脉冲 (+1.0)
        # 二阶导数防接飞刀铁律: 在极度鹰派区间(prev_sent <= -0.3)绝不接飞刀买入，必须等边际转向(sent_diff >= 0.15)即恐慌衰竭才买入
        # 或者无视常态，直接发生强烈的鸽派大转折(sent_diff >= 0.3)
        long_pulse = ((prev_sent <= -0.3) & (sent_diff >= 0.15)) | (sent_diff >= 0.3)
        
        # 2. 情绪恶化避险脉冲 (-1.0)
        # 在极度鸽派区间(prev_sent >= 0.3)，边际转坏(sent_diff <= -0.15)说明利好出尽趋势恶化
        # 或者直接发生强烈的鹰派突变恐慌(sent_diff <= -0.3)
        short_pulse = ((prev_sent >= 0.3) & (sent_diff <= -0.15)) | (sent_diff <= -0.3)
        
        pulse = pd.Series(0.0, index=data.index)
        pulse[long_pulse] = 1.0
        pulse[short_pulse] = -1.0
        
        # 零值休眠铁律: 常态必须返回 0.0。
        # FOMC会议每年约8次，如果仅在预期改变当天触发，Trigger Rate将极低(<3%)。
        # 严格遵守"极端事件发生的当天及随后极短几天内"的物理法则，我们将突变脉冲向后延展 3 天 (形成 4天 的狙击窗口)。
        # 假设每年有效触发 4~6 次，每次占据 4 天，则活跃交易日约为全年的 6.3% ~ 9.5%，完美落于 5%-15% 的铁律区间。
        signal = pulse.replace(0.0, np.nan).ffill(limit=3).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"