import numpy as np
import pandas as pd

class MicrostructureVixCurveCrossFactor:
    """微观结构恐慌衰竭与期限利差动量交叉因子 (microstructure/nonlinear)

    逻辑: 结合股市恐慌情绪衰竭与债市微观期限结构剧变的非线性交叉脉冲。当股市波动率(VIX)处于极端恐慌状态但开始衰竭，同时收益率曲线(T10Y2Y)出现瞬间的看涨变陡(Bull Steepening, 降息/避险预期急剧升温)，说明流动性冲击见顶且资金实质性回流美债，此时输出做多脉冲；反之做空。
    数据: vixcls, t10y2y
    触发: 多头 -> VIX 252日 Z-Score > 2.0 且 VIX < 3日均值 (衰竭) AND T10Y2Y 3日差值的 252日 Z-Score > 1.5 (剧烈变陡)。空头 -> VIX 极度贪婪反弹 + 曲线剧烈变平。
    输出: [-1.0, 1.0] 的极短期脉冲信号，常态休眠返回 0.0。
    """

    def __init__(self, vix_z_long=2.0, vix_z_short=-1.5, curve_mom_z=1.5, lookback=252, smooth=3):
        self.name = 'microstructure_vix_curve_cross_nonlinear'
        self.vix_z_long = vix_z_long
        self.vix_z_short = vix_z_short
        self.curve_mom_z = curve_mom_z
        self.lookback = lookback
        self.smooth = smooth

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖数据是否存在
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            return signal

        vix = data['vixcls'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # 计算 VIX Z-Score (识别恐慌/贪婪极值水位)
        vix_mean = vix.rolling(window=self.lookback, min_periods=self.lookback // 2).mean()
        vix_std = vix.rolling(window=self.lookback, min_periods=self.lookback // 2).std()
        vix_zscore = (vix - vix_mean) / (vix_std + 1e-8)

        # 铁律2: 二阶导数 - 必须伴随恐慌衰竭或贪婪反弹，严禁绝对水位直接接飞刀
        vix_ma = vix.rolling(window=self.smooth, min_periods=1).mean()
        vix_exhaustion = vix < vix_ma
        vix_rebound = vix > vix_ma

        # 铁律3: 边际变化 - 计算期限利差的动量瞬间变化，而非其绝对水位
        curve_diff = t10y2y.diff(self.smooth)
        curve_diff_mean = curve_diff.rolling(window=self.lookback, min_periods=self.lookback // 2).mean()
        curve_diff_std = curve_diff.rolling(window=self.lookback, min_periods=self.lookback // 2).std()
        curve_mom_zscore = (curve_diff - curve_diff_mean) / (curve_diff_std + 1e-8)

        # 交叉条件组合
        # 多头脉冲: 恐慌极值且见顶回落 + 曲线异常急剧变陡(避险情绪与宽松预期注入)
        buy_cond = (vix_zscore > self.vix_z_long) & vix_exhaustion & (curve_mom_zscore > self.curve_mom_z)
        
        # 空头脉冲: 贪婪极值且波动率开始反弹 + 曲线异常急剧变平(紧缩预期/抛售重燃)
        sell_cond = (vix_zscore < self.vix_z_short) & vix_rebound & (curve_mom_zscore < -self.curve_mom_z)

        # 铁律1: 零值休眠 - 只在极值事件+结构巨变时输出脉冲，其余时间严格为 0
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback={self.lookback}, smooth={self.smooth})"