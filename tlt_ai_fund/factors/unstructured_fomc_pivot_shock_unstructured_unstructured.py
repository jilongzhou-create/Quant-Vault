import numpy as np
import pandas as pd

class UnstructuredFomcPivotShockFactor:
    """Unstructured FOMC Policy Pivot Shock Factor (NLP Sentiment)

    逻辑: 捕捉美联储货币政策态度的极端反转(基于NLP提取的FOMC情绪得分)。美联储态度的突变是美债趋势反转的最强催化剂。由于该数据为低频阶梯跳跃特征，因子严格通过一阶差分计算边际突变，利用二阶差分寻找冲击衰竭点，确保仅在市场完全消化"惊吓"后才触发脉冲，避免在高波动首日接飞刀。
    数据: fomc_sentiment (FOMC声明鹰鸽情绪得分，范围[-1,1]，基于文本挖掘的阶梯状分布)
    触发: FOMC情绪3日变化量的 126日 Z-Score > 2.5 (极值)，且变化量的1日差分 <= 0 (二阶导数回落/衰竭)。
    输出: 脉冲型信号 [-1.0, 1.0]。鸽派突变且动能衰竭时输出 +1.0 (看多美债)，鹰派突变且动能衰竭时输出 -1.0 (看空美债)。常态绝对休眠为 0.0。
    """

    def __init__(self, window=126, z_threshold=2.5):
        self.name = 'unstructured_fomc_pivot_shock'
        # 126个交易日约为半年，覆盖3-4次FOMC会议周期，反映近期的政策态度基准
        self.window = window
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，默认全为0.0
        signal = pd.Series(0.0, index=data.index)

        # 检查数据完整性
        if 'fomc_sentiment' not in data.columns:
            return signal

        # 提取数据并前向填充，因为非会议日没有新值，保持阶梯状连续
        sentiment = data['fomc_sentiment'].ffill()

        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝对禁止使用绝对值直接触发。计算3日变化量，将阶梯跳跃转化为宽度为3天的脉冲窗口
        # 使用3天是为了平滑周末/假期的影响，并给后续的二阶衰竭留出计算空间
        delta_3d = sentiment.diff(3).fillna(0.0)

        # 计算滚动 Z-Score 衡量冲击的极端程度
        roll_mean = delta_3d.rolling(window=self.window, min_periods=21).mean()
        roll_std = delta_3d.rolling(window=self.window, min_periods=21).std()

        # 保护：因日常数据多为0，如果半年来没有大幅波动，std可能趋近于0，设置下限防止浮点数除法溢出
        roll_std = roll_std.clip(lower=1e-4)

        z_score = (delta_3d - roll_mean) / roll_std
        z_score = z_score.fillna(0.0)

        # 极端突变条件
        extreme_dovish = (z_score > self.z_threshold) & (delta_3d > 0)
        extreme_hawkish = (z_score < -self.z_threshold) & (delta_3d < 0)

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 绝对禁止在跳跃发生的第一天立即买入，必须等待加速度(二阶导)降温。
        # 计算变化量的1日差分(即情绪脉冲的加速度)。
        # 跳跃首日(T+1) accel 剧增；次日(T+2) delta_3d 仍处高位但 accel 降为 0，此时即为"动量平息/衰竭"的安全介入点。
        accel = delta_3d.diff(1).fillna(0.0)

        exhaustion_dovish = accel <= 0  # 鸽派冲击动能不再加速
        exhaustion_hawkish = accel >= 0 # 鹰派冲击动能不再加速向负向扩张

        # 组合触发逻辑
        buy_cond = extreme_dovish & exhaustion_dovish
        sell_cond = extreme_hawkish & exhaustion_hawkish

        # 仅在同时满足极值和衰竭条件时赋值
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold})"