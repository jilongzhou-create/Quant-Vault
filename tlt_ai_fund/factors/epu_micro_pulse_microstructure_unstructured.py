import numpy as np
import pandas as pd

class EpuMicroPulseFactor:
    """经济政策不确定性微观衰竭反转因子 (microstructure/unstructured)

    逻辑: 采用基于每日新闻文本(非结构化数据)衍生的经济政策不确定性(EPU)指数。当EPU短期急剧飙升反映市场宏观恐慌并达到极值，且随后见顶回落(边际缓解)时，输出做多美债脉冲；当EPU极度下降反映市场过度自满(Risk-On)并开始回升时，输出做空美债脉冲。常态下始终休眠输出0.0。
    数据: usepuindxd (Economic Policy Uncertainty Index, 文本NLP衍生)
    触发: 5日变化量的 252日 Z-Score > 2.5 且单日回落 (衰竭) -> +1.0；Z-Score < -2.5 且单日反弹 -> -1.0。
    输出: 极短期的狙击手级脉冲信号 (+1.0 或 -1.0)，持续3天，其余时间严格为 0.0。
    """

    def __init__(self):
        self.name = 'epu_micro_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理缺少必要字段的情况
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 提取指标并处理基础的缺失值
        epu = data['usepuindxd'].ffill()
        
        # 铁律3: 边际变化 Only (Marginal Change Only)
        # 绝对禁止使用指数的绝对水位，必须使用动量变化来捕捉预期的突变
        # 这里使用 5 日(一周)变化量作为短期势能
        epu_mom = epu.diff(5)
        
        # 计算势能的 252 日 (约1年) 滚动 Z-Score
        # min_periods=63 确保在累积一个季度数据后即可开始输出信号
        roll_mean = epu_mom.rolling(window=252, min_periods=63).mean()
        roll_std = epu_mom.rolling(window=252, min_periods=63).std()
        
        # 防止除以零
        roll_std = roll_std.replace(0.0, np.nan)
        epu_mom_z = (epu_mom - roll_mean) / roll_std

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 计算单日变化，用于判断极值是否开始衰竭
        daily_diff = epu.diff(1)
        
        # 条件1: 指标飙升至极端高位 (Z-Score > 2.5) 
        # 条件2: 指标开始回落，恐慌势能衰竭 (daily_diff < 0)
        long_trigger = (epu_mom_z > 2.5) & (daily_diff < 0)
        
        # 条件1: 指标暴跌至极端低位，极度自满 (Z-Score < -2.5)
        # 条件2: 指标跌无可跌开始反弹，自满势能衰竭 (daily_diff > 0)
        short_trigger = (epu_mom_z < -2.5) & (daily_diff > 0)

        # 铁律1: 零值休眠 (Sniper Pulse)
        # 初始默认全 0，常态下绝对不持有任何敞口
        signal = pd.Series(0.0, index=data.index)
        
        # 仅在触发日赋值
        signal.loc[long_trigger] = 1.0
        signal.loc[short_trigger] = -1.0

        # 将触发后的脉冲维持随后 2 天（共 3 天极短脉冲）以满足 5%-15% 的 Trigger Rate 要求
        signal = signal.replace(0.0, np.nan).ffill(limit=2).fillna(0.0)

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"