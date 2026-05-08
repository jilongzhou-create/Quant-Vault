import numpy as np
import pandas as pd

class YieldCurveMomentumPulseFactor:
    """政策预期冲量与曲线形态脉冲因子 (policy_pivot/nonlinear)

    逻辑: 捕捉前端利率异常下行与曲线急剧变陡(Bull Steepening)的极值交叉瞬间，反映市场在强烈交易流动性宽松预期，利多美股；反之短端飙升叠加曲线平坦化为紧缩恐慌，利空。
    数据: dgs2, t10y2y
    输出: +1.0 表示强烈看多(宽松预期突变)，-1.0 表示看空(紧缩恐慌突变)，0.0 处于常态休眠
    触发条件: 2年期国债收益率5日变动Z-Score和10Y2Y利差5日变动Z-Score处于异向极值区(|Z|>1.0)，仅在突破瞬间的第一天触发，预期 Trigger Rate 5%-15%
    """

    def __init__(self, window=5, z_window=252, z_threshold=1.0):
        self.name = 'yield_curve_momentum_pulse'
        self.window = window
        self.z_window = z_window
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理所需字段缺失的情况
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 数据前向填充，防止部分日期的缺失导致计算中断
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # 计算边际变化（动量）
        dgs2_diff = dgs2.diff(self.window)
        t10y2y_diff = t10y2y.diff(self.window)

        # 计算滚动的 Z-Score 反映边际变化的异常度
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(self.z_window).mean()) / dgs2_diff.rolling(self.z_window).std()
        t10y2y_z = (t10y2y_diff - t10y2y_diff.rolling(self.z_window).mean()) / t10y2y_diff.rolling(self.z_window).std()

        # 状态判定
        # Bull Steepening(牛陡): 短端异常下行且曲线异常变陡
        is_bull_steep = (dgs2_z < -self.z_threshold) & (t10y2y_z > self.z_threshold)
        
        # Bear Flattening(熊平): 短端异常上行且曲线异常平坦化/倒挂加深
        is_bear_flatten = (dgs2_z > self.z_threshold) & (t10y2y_z < -self.z_threshold)

        # 脉冲提取: 仅在状态改变(昨天不满足，今天满足)的瞬间触发
        bull_pulse = is_bull_steep & (~is_bull_steep.shift(1).fillna(False))
        bear_pulse = is_bear_flatten & (~is_bear_flatten.shift(1).fillna(False))

        # 零值休眠铁律构建
        signal = pd.Series(0.0, index=data.index)
        signal.loc[bull_pulse] = 1.0
        signal.loc[bear_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_window={self.z_window}, z_threshold={self.z_threshold})"