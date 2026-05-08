import numpy as np
import pandas as pd

class MicrostructureCurveVolNonlinearFactor:
    """Microstructure Curve Volatility Nonlinear Reversal Factor (microstructure/nonlinear)

    逻辑: 收益率曲线短端微观定价突变与全市场波动率衰竭的非线性交叉。当收益率曲线急剧陡峭化（避险资金抢筹短端引发的剧烈变陡）且VIX处于极值并开始衰竭时，确认恐慌见顶、微观流动性改善，产生做多长端美债的脉冲。反之在极度自满且曲线剧烈平坦化被打破时做空。
    数据: vixcls, t10y2y
    触发: VIX 252日 Z-Score > 2.5 且当日值 < 3日均值 (恐慌衰竭)，同时 T10Y2Y 5日动量的 252日 Z-Score > 2.0 (边际变陡突变) -> 输出 +1.0。
    输出: [-1.0, 1.0] 的多空脉冲信号，常态下严格保持 0.0。
    """

    def __init__(self, zscore_window=252, curve_window=5, vix_long_z=2.5, vix_short_z=-2.0, curve_z=2.0):
        self.name = 'microstructure_curve_vol_nonlinear'
        self.zscore_window = zscore_window
        self.curve_window = curve_window
        self.vix_long_z = vix_long_z
        self.vix_short_z = vix_short_z
        self.curve_z = curve_z

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        curve = data['t10y2y'].ffill()
        
        # 1. 波动率的极值与衰竭条件 (二阶导数铁律)
        vix_mean = vix.rolling(self.zscore_window).mean()
        vix_std = vix.rolling(self.zscore_window).std().replace(0, np.nan)
        vix_zscore = (vix - vix_mean) / vix_std
        
        vix_3d_mean = vix.rolling(3).mean()
        
        # 2. 收益率曲线微观定价动量 (边际变化铁律)
        # 不看绝对倒挂与否，只看短时间内边际变动的突发性 (动量的极端偏离)
        curve_momentum = curve.diff(self.curve_window)
        curve_mom_mean = curve_momentum.rolling(self.zscore_window).mean()
        curve_mom_std = curve_momentum.rolling(self.zscore_window).std().replace(0, np.nan)
        curve_mom_zscore = (curve_momentum - curve_mom_mean) / curve_mom_std
        
        # 3. 构造非线性交叉触发条件
        
        # 多头脉冲: 极值恐慌衰竭 + 剧烈陡峭化 (流动性危机见顶的标志组合)
        long_cond = (
            (vix_zscore > self.vix_long_z) &           # 条件1: 恐慌处于极端高位
            (vix < vix_3d_mean) &                      # 条件2: 恐慌开始衰竭 (高位折返)
            (curve_mom_zscore > self.curve_z) &        # 条件3: 曲线急剧陡峭化 (短端剧烈下行被抢筹)
            (curve.diff(1) > 0)                        # 条件4: 当日动能延续
        )
        
        # 空头脉冲: 自满情绪突发逆转 + 剧烈平坦化 (通胀或紧缩微观定价急剧升温)
        short_cond = (
            (vix_zscore < self.vix_short_z) &          # 条件1: 处于极度自满的极低位置
            (vix > vix_3d_mean) &                      # 条件2: 波动率开始突发反弹
            (curve_mom_zscore < -self.curve_z) &       # 条件3: 曲线急剧平坦化
            (curve.diff(1) < 0)                        # 条件4: 当日动能延续
        )
        
        # 零值休眠铁律：初始状态全0，仅脉冲日赋值
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, curve_window={self.curve_window})"