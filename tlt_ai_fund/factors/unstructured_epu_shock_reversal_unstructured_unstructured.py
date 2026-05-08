import numpy as np
import pandas as pd

class UnstructuredEpuShockReversalFactor:
    """经济政策不确定性恐慌衰竭脉冲 (unstructured/nlp)

    逻辑: 基于 NLP 新闻抓取的经济政策不确定性指数 (usepuindxd)。当不确定性飙升至极端高点且动量开始衰竭回落时，标志着避险情绪见顶和央行宽松预期的开启，此时输出看多美债(TLT)脉冲。当不确定性处于极度低迷（市场极度自满）且突然抬头时，预示着平静打破和通胀回归的压力，此时输出看空美债脉冲。
    数据: usepuindxd (每日经济政策不确定性指数)
    触发: 极值条件 (252日Z-Score > 2.5 或 < -2.0) 且 衰竭/反转条件 (EPU跌破或突破3日均线)。
    输出: +1.0 看多美债(避险爆发并衰竭), -1.0 看空美债(自满被打破)，严格短脉冲输出(单次触发仅维持3天)。
    """

    def __init__(self):
        self.name = 'unstructured_epu_shock_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始必须为全 0.0，严格遵守零值休眠铁律
        signal = pd.Series(0.0, index=data.index)

        if 'usepuindxd' not in data.columns:
            return signal

        # 处理可能的数据缺失
        epu = data['usepuindxd'].ffill()

        # 计算 252 日 (约1个交易年) 的移动均值和标准差，构建中长期相对极值坐标
        roll_mean = epu.rolling(window=252, min_periods=21).mean()
        roll_std = epu.rolling(window=252, min_periods=21).std() + 1e-6
        z_epu = (epu - roll_mean) / roll_std

        # 3日均线用于捕捉二阶导数：判断短期动量是否发生"衰竭"或"拐头"
        ma3_epu = epu.rolling(window=3, min_periods=1).mean()

        # ==========================================
        # 脉冲1: 看多美债 (恐慌见顶并衰竭)
        # ==========================================
        # 1. 绝对高位: 不确定性飙升进入极度恐慌
        cond_panic = z_epu > 2.5
        # 2. 二阶衰竭: 避免接飞刀，必须等待动量反转下破3日均线，确认情绪已过最高点
        cond_panic_exhaustion = epu < ma3_epu
        bull_event = cond_panic & cond_panic_exhaustion
        # 3. 边际变化: 捕捉刚刚完成反转的瞬间，拒绝连续发号
        bull_trigger = bull_event & (~bull_event.shift(1).fillna(False))

        # ==========================================
        # 脉冲2: 看空美债 (自满破灭被打破)
        # ==========================================
        # 1. 绝对低位: 不确定性极度低迷 (因EPU呈右偏分布，-2.0已属罕见的历史级低点)
        cond_complacency = z_epu < -2.0
        # 2. 抬头反转: 动量反转上破3日均线，标志着平静环境遭到破坏
        cond_complacency_break = epu > ma3_epu
        bear_event = cond_complacency & cond_complacency_break
        # 3. 边际变化: 捕捉刚刚抬头的瞬间
        bear_trigger = bear_event & (~bear_event.shift(1).fillna(False))

        # 为达到 5% ~ 15% 的目标 Trigger Rate，将触发信号通过 rolling 延展为事件发生后的极短 3 天脉冲
        bull_pulse = bull_trigger.rolling(window=3, min_periods=1).max().fillna(0) > 0
        bear_pulse = bear_trigger.rolling(window=3, min_periods=1).max().fillna(0) > 0

        # 赋值并生成最终脉冲信号
        signal.loc[bull_pulse] = 1.0
        signal.loc[bear_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"