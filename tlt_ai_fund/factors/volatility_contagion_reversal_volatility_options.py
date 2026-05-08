import numpy as np
import pandas as pd

class VolatilityContagionReversalFactor:
    """Volatility Contagion Reversal Factor (volatility/options)

    逻辑: 监控 CBOE 股票(VIX)与黄金(GVZ)期权隐含波动率的跨资产共振。当股金双杀、跨资产波动率极度飙升后出现同步回落时，标志着流动性恐慌(Margin Call)的极值点已过，避险资金将大规模涌入美债(TLT)。当波动率处于极度拥挤的低位(过度乐观)并开始同步向上发散时，标志着宏观稳态打破(如通胀或紧缩预期)，触发做空脉冲。因子严格遵循脉冲与衰竭验证，避免死于趋势主升/跌浪。
    数据: vixcls (SPX隐含波动率), gvzcls (黄金隐含波动率)
    触发: (联合 Z-Score > 1.5 AND 双双 diff < 0) -> +1.0; (联合 Z-Score < -1.0 AND 双双 diff > 0) -> -1.0
    输出: [-1.0, 1.0] 的脉冲信号，常态休眠为 0.0
    """

    def __init__(self, z_long_thresh=1.5, z_short_thresh=-1.0, window=252):
        self.name = 'volatility_contagion_reversal_options'
        self.z_long_thresh = z_long_thresh
        self.z_short_thresh = z_short_thresh
        self.window = window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 校验所需数据列是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal

        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()

        # 计算 252 日滚动的均值和标准差
        vix_mean = vix.rolling(window=self.window).mean()
        vix_std = vix.rolling(window=self.window).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std

        gvz_mean = gvz.rolling(window=self.window).mean()
        gvz_std = gvz.rolling(window=self.window).std().replace(0, np.nan)
        gvz_z = (gvz - gvz_mean) / gvz_std

        # 构建跨资产期权波动率联合压力指数 (联合 Z-Score)
        combo_z = (vix_z + gvz_z) / 2.0

        # 铁律3: 边际变化 (Marginal Change) - 计算动量变化捕捉拐点
        vix_diff = vix.diff()
        gvz_diff = gvz.diff()

        # 铁律2: 二阶导数验证 (Anti-Catch-Falling-Knife)
        # 看多条件：跨资产波动率极度狂飙 (Z > 1.5) AND 波动率开始同步衰竭回落 (diff < 0) -> 流动性冲击结束，买入美债
        long_cond = (combo_z > self.z_long_thresh) & (vix_diff < 0) & (gvz_diff < 0)

        # 看空条件：跨资产波动率极度低迷/拥挤 (Z < -1.0) AND 波动率开始同步爆发 (diff > 0) -> 风险平价策略解体，做空美债
        short_cond = (combo_z < self.z_short_thresh) & (vix_diff > 0) & (gvz_diff > 0)

        # 铁律1: 零值休眠 (Sniper Pulse) - 仅在极端衰竭日触发非零信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_long={self.z_long_thresh}, z_short={self.z_short_thresh}, window={self.window})"