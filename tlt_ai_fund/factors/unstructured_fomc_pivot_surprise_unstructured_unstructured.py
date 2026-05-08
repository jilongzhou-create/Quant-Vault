import numpy as np
import pandas as pd

class UnstructuredFomcPivotSurpriseFactor:
    """政策预期反转脉冲因子 (unstructured/unstructured)

    逻辑: 捕捉美联储前瞻指引情绪的极端突发转向。当NLP情绪得分发生逾2.5个标准差的巨大边际跳跃，且严格从鹰派区间反转跨越至鸽派区间时，形成"鹰转鸽衰竭反转"，输出看多脉冲；反之输出看空脉冲。天然通过差分几何特性形成完美的5日短脉冲，拒绝连续常态信号。
    数据: fomc_sentiment (基于LLM的FOMC鹰鸽得分, 阶梯数据)
    触发: 5日变化量 Z-Score > 2.5 (预期突变极值) + sentiment.shift(5)与当前值发生正负穿越 (二阶导数反转衰竭确认)
    输出: 狙击手级脉冲信号 [-1.0, 1.0], 仅在重大政策转向瞬间存续5天，其余时间严格为 0.0
    """

    def __init__(self):
        self.name = 'unstructured_fomc_pivot_surprise'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查依赖数据是否存在
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 针对阶梯数据，非会议日必须进行前向填充，以确保能在会议日正确计算跳跃差分
        sentiment = data['fomc_sentiment'].ffill()

        # 遵守铁律1：初始常态信号必须严格为 0.0 (零值休眠)
        signal = pd.Series(0.0, index=data.index)

        # 遵守铁律3：边际变化 (绝对禁止直接使用低频数据的绝对水位！)
        # 使用5日差分捕捉跳跃瞬间。由于阶梯特性，跳跃一旦发生，delta_5d 会在随后的5天内维持该跳跃幅度，自然形成5日交易脉冲
        delta_5d = sentiment.diff(5)

        # 计算边际变化的滚动 Z-Score (使用252日约1年窗口，衡量跳跃的极端程度)
        roll_mean = delta_5d.rolling(window=252, min_periods=60).mean()
        roll_std = delta_5d.rolling(window=252, min_periods=60).std()

        # 替换0值防止除零导致无限大或空值
        roll_std = roll_std.replace(0, np.nan)
        delta_z = (delta_5d - roll_mean) / roll_std

        # 遵守铁律2：二阶导数与衰竭 (绝对禁止无脑追涨，必须要求从相反的极端状态发生过零反转)
        # 鹰转鸽反转 (Dovish Pivot)：5天前明确处于偏鹰状态 (< 0)，当前突跳至偏鸽状态 (> 0)
        hawk_to_dove = (sentiment.shift(5) < 0.0) & (sentiment > 0.0)
        
        # 鸽转鹰反转 (Hawkish Pivot)：5天前明确处于偏鸽状态 (> 0)，当前突跳至偏鹰状态 (< 0)
        dove_to_hawk = (sentiment.shift(5) > 0.0) & (sentiment < 0.0)

        # 复合触发条件：突变极值 + 衰竭反转
        # 仅在产生罕见的结构性政策跳跃时，才发出单向买入/卖出狙击信号
        long_pulse = (delta_z > 2.5) & hawk_to_dove
        short_pulse = (delta_z < -2.5) & dove_to_hawk

        # 赋值正负看多看空信号 (美债 TLT 为 Carry 资产，降息鸽派看多，加息鹰派看空)
        signal[long_pulse] = 1.0
        signal[short_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"