import numpy as np
import pandas as pd

class YieldCurveMomentumShockFactor:
    """收益率曲线动量突变脉冲因子 (unstructured/nonlinear)

    逻辑: 捕捉由于美联储超预期政策转向导致的短端利率(DGS2)剧烈脉冲。纯粹基于边际变化，短端利率跳水且曲线牛陡时，视为强烈的降息预期骤生，脉冲看多TLT；短端利率飙升且曲线熊平时，视为加息/紧缩预期突发，脉冲看空TLT。
    数据: dgs2 (2年期国债收益率), t10y2y (10年-2年期限利差)
    触发: 
        看多(鸽派冲击): dgs2 5日变化量 Z-Score < -1.5 AND t10y2y 5日变化量 > 0 AND dgs2 < 3日均值 (确认下行动能)
        看空(鹰派冲击): dgs2 5日变化量 Z-Score > 1.5 AND t10y2y 5日变化量 < 0 AND dgs2 > 3日均值 (确认上行动能)
    输出: +1.0 (看多), -1.0 (看空), 非触发日严格为 0.0
    """

    def __init__(self, lookback: int = 252, diff_window: int = 5):
        self.name = 'yield_curve_momentum_shock'
        self.lookback = lookback
        self.diff_window = diff_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 异常与缺失列处理
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 数据前向填充，防止 NaN 破坏计算
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # 1. 边际变化铁律 (Marginal Change Only)
        # 禁止使用绝对水位，转为衡量 5 日维度的"突变动能"
        dgs2_diff = dgs2.diff(self.diff_window)
        t10y2y_diff = t10y2y.diff(self.diff_window)

        # 动态 Z-Score 评估极端事件
        dgs2_diff_mean = dgs2_diff.rolling(window=self.lookback, min_periods=self.diff_window).mean()
        dgs2_diff_std = dgs2_diff.rolling(window=self.lookback, min_periods=self.diff_window).std()
        
        # 避免除以 0 的极小值引发无穷大
        dgs2_diff_std = dgs2_diff_std.replace(0, np.nan)
        dgs2_z = (dgs2_diff - dgs2_diff_mean) / dgs2_diff_std

        # 2. 二阶导数衰竭铁律 (Anti-Catch-Falling-Knife)
        # 极端事件发生时，必须配合微观动能的确认，严禁趋势反转前的左侧接飞刀
        down_momentum = dgs2 < dgs2.rolling(3).mean()  # 收益率处于下行通道 (债市上涨)
        up_momentum = dgs2 > dgs2.rolling(3).mean()    # 收益率处于上行通道 (债市暴跌)

        # 3. 零值休眠铁律 (Sniper Pulse)
        signal = pd.Series(0.0, index=data.index)

        # 组合非线性交叉条件
        # 多头脉冲: 短端剧烈下行 (Z < -1.5) + 曲线牛陡 (短端下行快于长端) + 短期微观动能配合
        bull_steepening_pulse = (dgs2_z < -1.5) & (t10y2y_diff > 0) & down_momentum
        
        # 空头脉冲: 短端剧烈上行 (Z > 1.5) + 曲线熊平 (短端上行快于长端) + 短期微观动能配合
        bear_flattening_pulse = (dgs2_z > 1.5) & (t10y2y_diff < 0) & up_momentum

        # 信号赋值
        signal[bull_steepening_pulse] = 1.0
        signal[bear_flattening_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback={self.lookback}, diff_window={self.diff_window})"