import numpy as np
import pandas as pd

class FomcExtremumDivergenceFactor:
    """FOMC极端反转脉冲因子 (挖掘方向: policy_pivot, 方法: unstructured)

    逻辑: 捕捉政策周期绝对水位与边际动量的极值背离。市场常常被当下的高压政策水位锚定(极度悲观)，
          而在极度鹰派周期中出现的首次强烈的鸽派边际突变，意味着宏观紧缩周期实质性衰竭，催生强烈的看多脉冲("至暗曙光")。
          反之，在极度宽松的鸽派周期内突现大幅边际转鹰，意味着流动性盛宴临近尾声，催生看空脉冲("盛宴撤杯")。
    数据: fomc_sentiment (自然语言分析的FOMC会议态度得分)
    输出: +1.0 看多美股 (极度紧缩中出现鸽派大跳跃)
          -1.0 看空美股 (极度宽松中出现鹰派大跳跃)
    触发条件: T-1日水位超出门槛且T日边际跳跃幅度>0.15，动量发酵持续8个交易日，预期 Trigger Rate 约 5%-10%
    """

    def __init__(self, 
                 prev_hawkish_thresh: float = -0.1, 
                 prev_dovish_thresh: float = 0.1, 
                 dovish_jump_thresh: float = 0.15, 
                 hawkish_jump_thresh: float = -0.15,
                 pulse_window: int = 8):
        self.name = 'fomc_extremum_divergence_pulse'
        self.prev_hawkish_thresh = prev_hawkish_thresh
        self.prev_dovish_thresh = prev_dovish_thresh
        self.dovish_jump_thresh = dovish_jump_thresh
        self.hawkish_jump_thresh = hawkish_jump_thresh
        self.pulse_window = pulse_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        fomc = data['fomc_sentiment'].ffill()
        fomc_prev = fomc.shift(1)
        fomc_diff = fomc.diff()
        
        # 1. 甄别极端背离的突变日 (仅在FOMC阶梯数据发生边际变化的当天)
        # 买入条件: 处于鹰派紧缩周期且边际大幅度转鸽
        bull_trigger = (fomc_prev < self.prev_hawkish_thresh) & (fomc_diff > self.dovish_jump_thresh)
        # 卖出条件: 处于鸽派宽松周期且边际大幅度转鹰
        bear_trigger = (fomc_prev > self.prev_dovish_thresh) & (fomc_diff < self.hawkish_jump_thresh)
        
        # 2. 将低频的极值脉冲拉伸成可执行的短周期信号 (流动性冲击发酵期)
        bull_pulse = bull_trigger.rolling(window=self.pulse_window, min_periods=1).max()
        bear_pulse = bear_trigger.rolling(window=self.pulse_window, min_periods=1).max()
        
        # 3. 构造纯净的狙击手休眠信号
        signal = pd.Series(0.0, index=fomc.index)
        
        signal[bull_pulse > 0] = 1.0
        signal[bear_pulse > 0] = -1.0
        
        # 排除缺失和无效值影响
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(pulse_window={self.pulse_window})"