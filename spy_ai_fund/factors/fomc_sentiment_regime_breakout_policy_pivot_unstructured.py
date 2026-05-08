import numpy as np
import pandas as pd

class FomcSentimentRegimeBreakoutFactor:
    """FOMC情绪中枢突破因子 (policy_pivot/unstructured)

    逻辑: 捕捉 FOMC 声明情绪发生剧烈跳变并突破中长期均值中枢的极短窗口。使用 7 日差分计算边缘动量，利用低频阶梯数据的特性，这会天然构造一个持续约 7 个交易日的衰竭脉冲，使其在政策预期重估的一周内发出方向信号，随后自动休眠。
    数据: [fomc_sentiment]
    输出: 向鸽派大幅跳变且确立高于长期中枢输出 +1.0 (看多美股)，向鹰派大幅跳变且确立低于长期中枢输出 -1.0 (看空美股)。
    触发条件: 7天内发生绝对值超过 0.25 的剧烈转向且位于 120 天情绪均线的确认方向上，预期 Trigger Rate 5%-15%。
    """

    def __init__(self):
        self.name = 'fomc_sentiment_regime_breakout'
        # 阈值：0.25 代表情绪分数区间（-1 到 1，全距为2.0）发生了超过 12.5% 的显著边际突变
        self.jump_threshold = 0.25
        # 120个交易日约等于半年的自然时间，通常涵盖过去4次FOMC会议，作为中长期的基准预其中枢
        self.long_ema_span = 120
        # 7个交易日的动量窗口，保证每次信号触发后只会延续至多 7 天，满足脉冲铁律
        self.momentum_window = 7

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        fomc = data['fomc_sentiment'].ffill()
        
        # 计算7日边际动量。由于fomc_sentiment是阶梯状更新的低频数据，
        # 差分操作会在发生跳变后的连续 self.momentum_window 天内维持等于该跳变幅度，随后归零。
        # 这种巧妙特性可以防接飞刀，同时完美实现短窗口的脉冲延续。
        fomc_diff_7 = fomc.diff(self.momentum_window)
        
        # 计算过去120个交易日的情绪均值中枢
        fomc_ema_120 = fomc.ewm(span=self.long_ema_span, adjust=False).mean()
        
        # 多头脉冲：情绪剧烈转鸽（边际增量 >= 0.25）且当前得分确立于长期鸽派中枢之上
        bull_cond = (fomc_diff_7 >= self.jump_threshold) & (fomc > fomc_ema_120)
        
        # 空头脉冲：情绪剧烈转鹰（边际减量 <= -0.25）且当前得分确立于长期鹰派中枢之下
        bear_cond = (fomc_diff_7 <= -self.jump_threshold) & (fomc < fomc_ema_120)
        
        signal.loc[bull_cond] = 1.0
        signal.loc[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(jump_threshold={self.jump_threshold}, ema_span={self.long_ema_span})"