import numpy as np
import pandas as pd

class RatePulseSteepeningFactor:
    """流动性冲量与利率曲线脉冲因子 (policy_pivot/nonlinear)

    逻辑: 捕捉美联储政策预期的剧变瞬间。当短端利率(DGS2)在极短期内急剧下行且收益率曲线(T10Y2Y)快速变陡时，代表市场强烈抢跑降息(Bull Steepening)，形成看多的流动性冲量；反之，若短端利率急剧飙升且曲线变平/倒挂加深(Bear Flattening)，代表紧缩恐慌突袭，形成看空脉冲。
    数据: [dgs2, t10y2y]
    输出: 1.0 (流动性宽松冲量看多), -1.0 (紧缩冲击看空), 0.0 (常态休眠)
    触发条件: DGS2 5日变动 > 15bps 与 T10Y2Y 5日变动 > 8bps 进行非线性交叉，极值脉冲维持3天，预期 Trigger Rate 在 8%-12% 左右。
    """

    def __init__(self):
        self.name = 'rate_pulse_steepening'
        # 参数具有严格经济学含义：5个交易日(1周)的短端突变
        self.lookback = 5
        self.dgs2_threshold = 0.15      # 15个基点(快速定价半次FOMC变动预期)
        self.t10y2y_threshold = 0.08    # 8个基点(曲线形态动量的剧烈变化)
        self.pulse_hold_days = 3        # 极短几天的脉冲窗口

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 边际变化铁律：捕捉预期改变瞬间的动量变化，禁止使用绝对水位
        dgs2_diff = dgs2.diff(self.lookback)
        t10y2y_diff = t10y2y.diff(self.lookback)
        
        # 狙击手条件1：抢跑宽松 (Bull Steepening)
        # 短期内短端利率猛烈下行，且曲线急剧变陡
        bull_cond = (dgs2_diff <= -self.dgs2_threshold) & (t10y2y_diff >= self.t10y2y_threshold)
        
        # 狙击手条件2：紧缩恐慌 (Bear Flattening)
        # 短期内短端利率急剧飙升，且曲线走平/倒挂加深
        bear_cond = (dgs2_diff >= self.dgs2_threshold) & (t10y2y_diff <= -self.t10y2y_threshold)
        
        # 零值休眠铁律：利用 rolling.max() 将脉冲信号仅维持极短的3天
        bull_pulse = bull_cond.rolling(window=self.pulse_hold_days, min_periods=1).max() == 1
        bear_pulse = bear_cond.rolling(window=self.pulse_hold_days, min_periods=1).max() == 1
        
        # 合成最终信号
        signal[bull_pulse] = 1.0
        signal[bear_pulse & ~bull_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback={self.lookback}, pulse_days={self.pulse_hold_days})"