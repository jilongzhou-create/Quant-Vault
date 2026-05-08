import numpy as np
import pandas as pd

class FomcExtremesCapitulationPulseFactor:
    """极端情绪投降脉冲因子 (policy_pivot/unstructured)

    逻辑: 捕捉FOMC预期在"极端一致"时的突然边际逆转(Capitulation)。美股不怕一直鹰，就怕在极度乐观时突然泼冷水；也不怕一直鸽，而在政策预期极度压抑(至暗时刻)时，任何边际放松都会点燃巨大的多头动能。
    数据: [fomc_sentiment]
    输出: 边际转鸽看多(+1.0)，边际转鹰看空(-1.0)，平时休眠(0.0)。
    触发条件: 上期情绪处于极端区(绝对值>0.15) 且 本期发生逆向跳变(差分>0.10)。市场需要消化宏观拐点，因此信号持续10个交易日，预期Trigger Rate在 5% 到 15% 之间。
    """

    def __init__(self, hawk_threshold=-0.15, dove_threshold=0.15, diff_threshold=0.10, pulse_window=10):
        self.name = 'fomc_extremes_capitulation_pulse'
        # 鹰/鸽极端情绪判定水位 (经济学含义: 强烈的单边宏观预期)
        self.hawk_threshold = hawk_threshold
        self.dove_threshold = dove_threshold
        # 边际变化阈值 (经济学含义: 足以打破原有趋势预期的边际外力冲击)
        self.diff_threshold = diff_threshold
        # 市场定价消化期 (经济学含义: 机构投资者调整头寸与配置通常需要一到两周)
        self.pulse_window = pulse_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 数据缺失校验
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        sentiment = data['fomc_sentiment'].ffill()
        # 获取发生变化前一天的情绪水位
        prev_sentiment = sentiment.shift(1)
        # 提取边际变化量
        diff = sentiment.diff()
        
        # 阶梯状数据只有在更新日(diff != 0)才是真实的FOMC会议窗口
        is_update = diff.abs() > 0.001
        
        # 多头触发 (Hawkish Capitulation): 极度压抑的鹰派预期下，美联储突然边际转鸽
        bull_trigger = is_update & (prev_sentiment < self.hawk_threshold) & (diff > self.diff_threshold)
        
        # 空头触发 (Dovish Capitulation): 极度乐观的鸽派预期下，美联储突然边际转鹰
        bear_trigger = is_update & (prev_sentiment > self.dove_threshold) & (diff < -self.diff_threshold)
        
        signal = pd.Series(0.0, index=data.index)
        signal.loc[bull_trigger] = 1.0
        signal.loc[bear_trigger] = -1.0
        
        # 将脉冲维持一个短期的消化窗口(pulse_window)，其余时间维持0.0休眠
        # 使用 replace(0.0, np.nan) 后 ffill 再 fillna(0.0) 形成脉冲拖尾
        signal = signal.replace(0.0, np.nan).ffill(limit=self.pulse_window - 1).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(hawk_threshold={self.hawk_threshold}, dove_threshold={self.dove_threshold}, diff_threshold={self.diff_threshold}, pulse_window={self.pulse_window})"