import numpy as np
import pandas as pd

class YieldCurveSteepeningPulseFactor:
    """收益率曲线牛陡流动性脉冲因子 (policy_pivot/nonlinear)

    逻辑: 当短端利率(DGS2)在短时间内剧烈下行，且收益率曲线(T10Y2Y)急剧变陡时，表明市场正在强烈抢跑美联储降息预期(Bull Steepening)，流动性预期发生鸽派突变，这是强烈的看多冲量；反之，若短端利率剧烈上升导致曲线加速变平(Bear Flattening)，则是紧缩预期的突变，看空美股。
    数据: [t10y2y, dgs2]
    输出: 1.0 看多 (牛市陡峭化，降息预期), -1.0 看空 (熊市平坦化，加息预期)
    触发条件: DGS2 5日跌幅超15bps 且 T10Y2Y 5日变陡超10bps触发看多，反之看空。使用严格的边际变化脉冲控制，预期 Trigger Rate 5%-10%。
    """

    def __init__(self, lookback_window: int = 5, dgs2_thresh: float = 0.15, t10y2y_thresh: float = 0.10):
        self.name = 'yield_curve_steepening_pulse'
        # 考察期：通常一周内的预期剧变
        self.lookback_window = lookback_window
        # 短端利率变动阈值：15个基点(bps)，反映相当于半次以上加降息的预期重估
        self.dgs2_thresh = dgs2_thresh
        # 期限利差变动阈值：10个基点(bps)，确认长短端出现分化，曲线发生形变
        self.t10y2y_thresh = t10y2y_thresh

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 't10y2y' not in data.columns or 'dgs2' not in data.columns:
            return signal
            
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 计算利率市场的边际动量变化 (单位：%)
        dgs2_diff = dgs2.diff(self.lookback_window)
        t10y2y_diff = t10y2y.diff(self.lookback_window)
        
        # 非线性交叉 1：牛市陡峭化 (看多)
        # 逻辑：短端剧烈下行，说明流动性预期极度转鸽；且幅度大于长端导致曲线变陡
        bull_steepening = (dgs2_diff < -self.dgs2_thresh) & (t10y2y_diff > self.t10y2y_thresh)
        
        # 非线性交叉 2：熊市平坦化 (看空)
        # 逻辑：短端剧烈上行，紧缩预期重燃；且幅度大于长端导致曲线倒挂加深/变平
        bear_flattening = (dgs2_diff > self.dgs2_thresh) & (t10y2y_diff < -self.t10y2y_thresh)
        
        # 脉冲化过滤：零值休眠铁律
        # 只在预期剧变首次达标的瞬间(发生jump的第一天)触发信号，拒绝在状态延续期间连续输出信号
        bull_pulse = bull_steepening & (~bull_steepening.shift(1).fillna(False))
        bear_pulse = bear_flattening & (~bear_flattening.shift(1).fillna(False))
        
        signal[bull_pulse] = 1.0
        signal[bear_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback={self.lookback_window}, dgs2_th={self.dgs2_thresh}, t10y_th={self.t10y2y_thresh})"