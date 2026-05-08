import numpy as np
import pandas as pd

class UnstructuredGoldVolExhaustionFactor:
    """Unstructured Gold Volatility Exhaustion (unstructured/options)

    逻辑: 黄金隐含波动率(GVZ)是期权市场对恶性通胀和地缘尾部风险定价的前瞻指标。当GVZ飙升至极端高位并开始回落时，标志着通胀恐慌和极度避险情绪的衰竭(Peak Panic)，宏观利率上行压力骤减，美债(TLT)迎来绝佳的右侧做多脉冲；反之，若波动率极度低迷后突然抬头，预示新一轮通胀/避险冲击，做空美债。
    数据: gvzcls (CBOE黄金ETF隐含波动率指数)
    触发: GVZ 252日 Z-Score > 2.5 且开始衰竭(单日下跌且低于3日均值) -> +1.0；Z-Score < -2.0 且边际抬升 -> -1.0
    输出: [-1.0, 0.0, 1.0] 的狙击手级脉冲信号
    """

    def __init__(self, window=252, z_long=2.5, z_short=-2.0, smooth_window=3):
        self.name = 'unstructured_gold_vol_exhaustion'
        self.window = window
        self.z_long = z_long
        self.z_short = z_short
        self.smooth_window = smooth_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号全为0，严格遵守零值休眠铁律
        signal = pd.Series(0.0, index=data.index)

        # 缺失字段处理
        if 'gvzcls' not in data.columns:
            return signal

        # 提取数据并处理潜在的NaN (使用ffill避免引入未来数据)
        gvz = data['gvzcls'].ffill()

        # 计算 252日(约一年) 的滚动 Z-Score 衡量极端水位
        # 使用 min_periods 确保计算的稳定性
        roll_mean = gvz.rolling(window=self.window, min_periods=self.window // 2).mean()
        roll_std = gvz.rolling(window=self.window, min_periods=self.window // 2).std()
        
        # 避免除以0的情况
        roll_std = roll_std.replace(0.0, np.nan)
        zscore = (gvz - roll_mean) / roll_std

        # 计算边际变化，遵守二阶导数铁律(衰竭确认)
        gvz_diff = gvz.diff()
        gvz_ma = gvz.rolling(window=self.smooth_window).mean()

        # 多头脉冲: 极度恐慌 (Z > 2.5) + 开始衰竭 (动量转负 且 下穿均线) -> 防接飞刀
        long_cond = (zscore > self.z_long) & (gvz_diff < 0) & (gvz < gvz_ma)

        # 空头脉冲: 极度自满 (Z < -2.0) + 边际爆发 (动量转正 且 上穿均线)
        short_cond = (zscore < self.z_short) & (gvz_diff > 0) & (gvz > gvz_ma)

        # 触发信号赋值
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"UnstructuredGoldVolExhaustionFactor(window={self.window}, z_long={self.z_long}, z_short={self.z_short})"