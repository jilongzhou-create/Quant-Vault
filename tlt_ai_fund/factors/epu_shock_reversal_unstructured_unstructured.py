import numpy as np
import pandas as pd

class EpuShockReversalFactor:
    """经济政策不确定性突变与衰竭脉冲因子 (unstructured/unstructured)

    逻辑: EPU(经济政策不确定性)基于主流新闻的NLP提取, 反映市场对宏观政策的恐慌度。当 EPU 边际暴涨至极端值时, 大量资金涌入美债避险; 一旦暴涨动能衰竭(靴子落地), 避险盘撤出导致美债回落(看空)。反之, 当 EPU 暴跌至极端且衰竭时, 极致 Risk-On 情绪见顶, 资金将重新回补美债(看多)。常态下信号必须休眠。
    数据: usepuindxd (美国经济政策不确定性日度指数)
    触发: 5日变化量的 252日 Z-Score > 2.5 且边际变化小于3日均值(恐慌见顶衰竭) -> -1.0; Z-Score < -2.5 且边际变化大于3日均值(狂热见顶衰竭) -> +1.0
    输出: -1.0 (恐慌退潮看空美债) 或 +1.0 (狂热退潮看多美债) 的脉冲信号
    """

    def __init__(self, diff_window=5, zscore_window=252, exhaust_window=3, z_threshold=2.5):
        self.name = 'epu_shock_reversal_factor'
        self.diff_window = diff_window
        self.zscore_window = zscore_window
        self.exhaust_window = exhaust_window
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (Sniper Pulse)
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only) - 绝对禁止使用绝对值
        epu_diff = epu.diff(self.diff_window)
        
        # 计算 252个交易日 (一年) 维度的滚动 Z-Score
        roll_mean = epu_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).mean()
        roll_std = epu_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).std()
        
        # 防止除零引发的魔法错误
        roll_std = roll_std.replace(0.0, np.nan)
        zscore = (epu_diff - roll_mean) / roll_std
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife) - 动能必须衰竭
        diff_mean = epu_diff.rolling(window=self.exhaust_window, min_periods=1).mean()
        
        # 恐慌飙升 (Z-Score > 2.5) 且 向上动能开始衰竭 (当前边际小于最近3日平均水平)
        up_exhausted = (zscore > self.z_threshold) & (epu_diff < diff_mean)
        
        # 狂热暴跌 (Z-Score < -2.5) 且 向下动能开始衰竭 (当前边际大于最近3日平均水平)
        down_exhausted = (zscore < -self.z_threshold) & (epu_diff > diff_mean)
        
        # 脉冲触发赋值
        signal.loc[up_exhausted] = -1.0  # 恐慌消退，避险资金撤出，看空美债
        signal.loc[down_exhausted] = 1.0 # 极度乐观消退，避险资金回流，看多美债
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(diff_window={self.diff_window}, zscore_window={self.zscore_window}, exhaust_window={self.exhaust_window}, z_threshold={self.z_threshold})"