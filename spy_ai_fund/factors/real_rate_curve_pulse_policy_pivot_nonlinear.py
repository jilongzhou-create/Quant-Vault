import numpy as np
import pandas as pd

class RealRateCurvePulseFactor:
    """实际利率与收益率曲线共振的政策转向脉冲因子 (policy_pivot/nonlinear)

    逻辑: 结合长端实际利率(DFII10)和短端名义利率(DGS2)的剧烈边际变化。当市场抢跑降息时，短端利率暴跌带动曲线急剧变陡(Bull Steepening)，同时长端实际利率急剧下行(贴现率下降)，产生看多美股的流动性冲量；反之，鹰派惊吓导致短端急升、曲线倒挂加深及实际利率飙升，产生看空美股的脉冲。
    数据: dgs2, t10y2y, dfii10
    输出: +1.0 (流动性预期大幅转松，强烈看多), -1.0 (流动性预期急剧收紧，看空)
    触发条件: 5个交易日内，2年期收益率下跌>15bps，实际利率下跌>10bps，利差扩大>8bps瞬间触发看多；反之亦然。预期Trigger Rate控制在5-15%左右。
    """

    def __init__(
        self,
        lookback_window: int = 5,
        dgs2_change_bps: float = 0.15,
        t10y2y_change_bps: float = 0.08,
        dfii10_change_bps: float = 0.10
    ):
        self.name = 'real_rate_curve_pulse_nonlinear'
        self.lookback_window = lookback_window
        self.dgs2_change_bps = dgs2_change_bps
        self.t10y2y_change_bps = t10y2y_change_bps
        self.dfii10_change_bps = dfii10_change_bps

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index, name=self.name)

        required_cols = ['dgs2', 't10y2y', 'dfii10']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        # 填补国债假日的缺失值以保持时间序列连贯性
        df = data[required_cols].ffill()
        
        # 计算窗口期内的边际动量变化
        dgs2_diff = df['dgs2'].diff(self.lookback_window)
        t10y2y_diff = df['t10y2y'].diff(self.lookback_window)
        dfii10_diff = df['dfii10'].diff(self.lookback_window)
        
        # 鸽派突变 (流动性极剧转松)
        # 逻辑: 短端收益率暴跌 + 曲线急速变陡 + 实际贴现率显著下行
        bull_steepening = (
            (dgs2_diff <= -self.dgs2_change_bps) & 
            (t10y2y_diff >= self.t10y2y_change_bps) & 
            (dfii10_diff <= -self.dfii10_change_bps)
        )
        
        # 鹰派惊吓 (流动性极剧转紧)
        # 逻辑: 短端收益率飙升 + 倒挂加深/曲线平坦化 + 实际贴现率显著上行
        bear_flattening = (
            (dgs2_diff >= self.dgs2_change_bps) & 
            (t10y2y_diff <= -self.t10y2y_change_bps) & 
            (dfii10_diff >= self.dfii10_change_bps)
        )
        
        signal[bull_steepening] = 1.0
        signal[bear_flattening] = -1.0
        
        return signal

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"lookback_window={self.lookback_window}, "
                f"dgs2_change_bps={self.dgs2_change_bps}, "
                f"t10y2y_change_bps={self.t10y2y_change_bps}, "
                f"dfii10_change_bps={self.dfii10_change_bps})")