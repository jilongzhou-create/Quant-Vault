import numpy as np
import pandas as pd

class VixGvzPanicExhaustionFactor:
    """微观结构/期权 (microstructure/options) 跨资产波动率恐慌衰竭脉冲因子

    逻辑: 股票恐慌(VIX)与黄金避险恐慌(GVZ)的差值衡量了系统性风险偏好的极度失衡。当差值达到历史极高位(Z-Score>2.5)时，代表股票市场发生流动性危机引发极致抛售；当该差值见顶回落时，标志恐慌情绪衰竭，避险资金重新回流长端美债，形成做多TLT脉冲。反之滞胀恐慌时做空。
    数据: vixcls, gvzcls
    触发: (VIX - GVZ) 252日 Z-Score > 2.5 且 当前差值 < 过去3日均值 (衰竭反转)
    输出: +1.0 (恐慌见顶做多脉冲), -1.0 (滞胀恐慌做空脉冲)
    """

    def __init__(self, window=252, z_threshold=2.5, smooth_window=3):
        self.name = 'microstructure_options_vix_gvz_skew_exhaustion'
        self.window = window
        self.z_threshold = z_threshold
        self.smooth_window = smooth_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal

        # 提取数据并处理缺失值
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()

        # 波动率差值 (跨资产波动率倾斜: 股市恐慌 vs 黄金恐慌)
        vol_skew = vix - gvz

        # 计算 252日 Z-Score (边际极值)
        roll_mean = vol_skew.rolling(window=self.window, min_periods=self.window // 2).mean()
        roll_std = vol_skew.rolling(window=self.window, min_periods=self.window // 2).std()
        
        z_score = (vol_skew - roll_mean) / roll_std.replace(0, np.nan)

        # 多头触发条件: 股票极度恐慌 + 开始衰竭回落
        extreme_panic = z_score > self.z_threshold
        exhaustion_down = vol_skew < vol_skew.rolling(window=self.smooth_window).mean()
        buy_cond = extreme_panic & exhaustion_down

        # 空头触发条件: 黄金隐含波动率远超股票(典型滞胀/突发通胀恐慌) + 开始边际反转
        extreme_inflation_panic = z_score < -self.z_threshold
        exhaustion_up = vol_skew > vol_skew.rolling(window=self.smooth_window).mean()
        sell_cond = extreme_inflation_panic & exhaustion_up

        # 生成脉冲信号
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold}, smooth_window={self.smooth_window})"