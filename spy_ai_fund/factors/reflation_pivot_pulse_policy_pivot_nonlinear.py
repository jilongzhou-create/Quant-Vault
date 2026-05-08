import numpy as np
import pandas as pd

class ReflationPivotPulseFactor:
    """Reflation Pivot Pulse Factor (policy_pivot/nonlinear)

    逻辑: 捕捉实际利率(DFII10)极端跳水与通胀预期(T10YIE)企稳反弹的高维非线性交叉。当实际利率暴跌(政策释放极度宽松冲量)且通胀预期未随之恶化时，表明市场正在快速定价'鸽派转向+软着陆'，这对美股极度利好；反之，若实际利率极端飙升且通胀/增长预期恶化，则为鹰派政策失误导致的紧缩恐慌，对美股极度利空。
    数据: dfii10 (10年期实际利率), t10yie (10年期盈亏平衡通胀预期)
    输出: 脉冲信号，[-1.0, 1.0]。+1.0为软着陆宽松预期剧变，-1.0为紧缩恐慌剧变。常态返回 0.0。
    触发条件: 实际利率5日动量的Z-Score < -1.5 且 通胀预期5日动量的Z-Score > 0.0 时触发多头。双向总体预期Trigger Rate控制在 5%-15% 之间。
    """

    def __init__(self, momentum_window=5, zscore_window=252, extreme_z=1.5):
        self.name = 'reflation_pivot_pulse_nonlinear'
        self.momentum_window = momentum_window # 5个交易日捕捉低频预期的边际突变窗口
        self.zscore_window = zscore_window     # 252个交易日(约1年)适应不同宏观周期的波动率基准
        self.extreme_z = extreme_z             # 1.5 Z-Score 对应统计学上单侧约6.7%的尾部极值概率

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 零值休眠铁律: 遇到缺失数据也必须安全返回全0信号
        if 'dfii10' not in data.columns or 't10yie' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 预处理数据: 前向填充，防止非交易日的扰动
        ry = data['dfii10'].ffill()
        be = data['t10yie'].ffill()

        # 边际变化铁律: 绝对禁止使用绝对水位，必须使用.diff()计算动量跳跃
        ry_diff = ry.diff(self.momentum_window)
        be_diff = be.diff(self.momentum_window)

        # 计算滚动Z-Score，识别动量极值状态
        ry_mean = ry_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).mean()
        ry_std = ry_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).std()
        
        be_mean = be_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).mean()
        be_std = be_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).std()

        # 安全计算 Z-Score (避免除以0引发的无限大)
        ry_z = (ry_diff - ry_mean) / ry_std.replace(0, np.nan)
        be_z = (be_diff - be_mean) / be_std.replace(0, np.nan)

        # 初始化狙击手零值信号
        signal = pd.Series(0.0, index=data.index, name=self.name)

        # 特征交叉 (纯粹基于同利率维度的经济学交叉)
        # 看多买点(Goldilocks): 实际利率极端跳水（美联储救市/流动性大放水） + 盈亏平衡通胀预期未跌或反弹（经济基本面未崩盘）
        bull_cond = (ry_z < -self.extreme_z) & (be_z > 0.0)
        
        # 看空买点(Stagflation/Hawkish Error): 实际利率极端飙升（流动性恐慌收紧） + 盈亏平衡预期跌破均值（硬着陆担忧）
        bear_cond = (ry_z > self.extreme_z) & (be_z < 0.0)

        # 仅在数据均已就绪且符合脉冲事件特征的当天输出信号
        valid_idx = ry_z.notna() & be_z.notna()
        
        signal.loc[bull_cond & valid_idx] = 1.0
        signal.loc[bear_cond & valid_idx] = -1.0

        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(momentum_window={self.momentum_window}, zscore_window={self.zscore_window}, extreme_z={self.extreme_z})"