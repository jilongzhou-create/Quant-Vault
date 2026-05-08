import numpy as np
import pandas as pd

class FomcPeakReversionPulseFactor:
    """FOMC极值反转脉冲因子 (policy_pivot/unstructured)

    逻辑: 捕捉美联储极端鹰派或鸽派情绪衰竭后的首次边际逆转。当上一次会议情绪处于极度鹰派水位，且本次会议声明发生明显的鸽派跳升(逆转)时，确认紧缩周期见顶并形成看多脉冲；反之亦然。信号向后延续极短的几天以覆盖资产重新定价窗口。
    数据: fomc_sentiment
    输出: 1.0表示美联储鹰派见顶转鸽看多美股，-1.0表示鸽派见顶转鹰看空美股。常态为0.0。
    触发条件: 绝对情绪水位偏向极端(>=0.15或<=-0.15)且边际跳跃幅度明显(>=0.2)，信号保持6个交易日。预期Trigger Rate处于5%-15%区间内。
    """

    def __init__(self, hawkish_threshold: float = -0.15, dovish_threshold: float = 0.15, diff_threshold: float = 0.20, hold_days: int = 6):
        self.name = 'fomc_peak_reversion_pulse'
        self.hawkish_threshold = hawkish_threshold
        self.dovish_threshold = dovish_threshold
        self.diff_threshold = diff_threshold
        self.hold_days = hold_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        # FOMC情绪得分是低频阶梯状数据，需要前向填充
        sentiment = data['fomc_sentiment'].ffill()
        
        # 计算动量变化(阶梯跳跃幅度)
        diff = sentiment.diff()
        prev_sentiment = sentiment.shift(1)
        
        # 多头脉冲: 极值(鹰派环境) + 衰竭逆转(突发鸽派跳升)
        bull_trigger = (prev_sentiment <= self.hawkish_threshold) & (diff >= self.diff_threshold)
        
        # 空头脉冲: 极值(鸽派环境) + 衰竭逆转(突发鹰派跳升)
        bear_trigger = (prev_sentiment >= self.dovish_threshold) & (diff <= -self.diff_threshold)
        
        raw_signal = pd.Series(0.0, index=data.index)
        raw_signal[bull_trigger] = 1.0
        raw_signal[bear_trigger] = -1.0
        
        # 零值休眠铁律 & 脉冲向后延伸:
        # 发生突变的当天及随后的 hold_days 天内保持非零脉冲信号，以模拟市场对 Pivot 预期的定价消化期
        bull_signal = raw_signal.where(raw_signal == 1.0, 0.0).rolling(window=self.hold_days, min_periods=1).max()
        bear_signal = raw_signal.where(raw_signal == -1.0, 0.0).rolling(window=self.hold_days, min_periods=1).min()
        
        signal = bull_signal + bear_signal
        
        # 清理NaN并将最终信号限制在 [-1.0, 1.0]
        signal = signal.fillna(0.0).clip(-1.0, 1.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(hawkish_threshold={self.hawkish_threshold}, dovish_threshold={self.dovish_threshold}, diff_threshold={self.diff_threshold}, hold_days={self.hold_days})"