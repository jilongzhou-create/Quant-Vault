import numpy as np
import pandas as pd

class UnstructuredSentimentUncertaintyCrossFactor:
    """Unstructured Sentiment & Uncertainty Cross Factor (unstructured/nonlinear)

    逻辑: 捕捉经济政策不确定性(USEPU)的恐慌极值衰竭与美联储(FOMC)情绪边际突变的非线性交叉。
          作为狙击手级卫星因子，本策略在两种情况下触发脉冲信号: 
          1. 政策预期突变 (Policy Pivot Shock): FOMC情绪得分的边际变化出现超2.5个标准差的鸽派/鹰派跳跃;
          2. 非线性特征交叉 (Nonlinear Feature Cross): 在FOMC中期动量偏鸽(鹰)的背景过滤下，政策不确定性指数爆发超2.5个标准差的恐慌极值且当日开始衰竭回落(或极度自满被打破)。
          该逻辑严格遵守零值休眠、二阶导数衰竭(避免接飞刀)以及只看边际变化的三大铁律。
    数据: usepuindxd (经济政策不确定性指数), fomc_sentiment (FOMC鹰鸽情绪得分)
    触发: (USEPU Z-Score > 2.5 且开始衰竭 且 FOMC中期动量偏鸽) OR (FOMC边际突变 Z-Score > 2.5) -> +1.0
    输出: 狙击手级别的脉冲信号 [-1.0, 1.0]，非触发日常态严格为 0.0
    """

    def __init__(self):
        self.name = 'unstructured_sentiment_uncertainty_cross_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化常态为 0.0 的脉冲 Series
        signal = pd.Series(0.0, index=data.index)

        # 校验所需数据列是否缺失
        if 'usepuindxd' not in data.columns or 'fomc_sentiment' not in data.columns:
            return signal

        usepu = data['usepuindxd'].ffill()
        fomc = data['fomc_sentiment'].ffill()

        # --------------------------------------------------------------------------------
        # 1. 经济政策不确定性脉冲 (恐慌衰竭与自满打破)
        # --------------------------------------------------------------------------------
        # 提取短期局部脉冲，使用 63 日(单季度)滚动窗口计算 Z-Score，保留肥尾事件
        usepu_mean = usepu.rolling(window=63).mean()
        usepu_std = usepu.rolling(window=63).std() + 1e-6
        usepu_z = (usepu - usepu_mean) / usepu_std
        usepu_diff = usepu.diff(1).fillna(0)

        # 铁律2 (二阶导数): 包含恐慌指标必须加入衰竭条件
        # 看多条件基础: 极度恐慌 (Z > 2.5) + 开始回落 (diff < 0) -> 不确定性达峰衰竭
        usepu_peak_exhaustion = (usepu_z > 2.5) & (usepu_diff < 0)
        
        # 看空条件基础: 极度自满 (Z < -2.5) + 开始攀升 (diff > 0) -> 风险萌芽
        usepu_bottom_breakout = (usepu_z < -2.5) & (usepu_diff > 0)

        # --------------------------------------------------------------------------------
        # 2. FOMC 情绪得分边际突变与中期动量
        # --------------------------------------------------------------------------------
        # 铁律3 (边际变化): 绝对禁止直接输出绝对值，必须使用 diff 或动量
        
        # A. 中期动量 (用于特征交叉的背景过滤) - 对比半年度(126日)均值
        fomc_ma126 = fomc.rolling(window=126).mean()
        fomc_momentum = (fomc - fomc_ma126).fillna(0)

        # B. 短期边际突变 (捕捉 FOMC 决议日当天的跳跃)
        # 使用 diff(3) 吸收会议前后的预热和落地
        fomc_diff3 = fomc.diff(3).fillna(0)
        fomc_diff_mean = fomc_diff3.rolling(window=252).mean()
        fomc_diff_std = fomc_diff3.rolling(window=252).std() + 1e-6
        fomc_z = (fomc_diff3 - fomc_diff_mean) / fomc_diff_std

        # 突变条件: 严格执行 Z-Score > 2.5 的标准
        fomc_dovish_shock = fomc_z > 2.5   # 鸽派突变
        fomc_hawkish_shock = fomc_z < -2.5 # 鹰派突变

        # --------------------------------------------------------------------------------
        # 3. 信号综合 (非线性特征交叉)
        # --------------------------------------------------------------------------------
        # 正向脉冲 (看多美债 TLT):
        # 触发点 A: 央行发生极端的鸽派转向突变
        # 触发点 B: 在央行总体偏鸽的背景下，发生极端的宏观不确定性恐慌，且恐慌已经开始衰竭
        long_cond = fomc_dovish_shock | (usepu_peak_exhaustion & (fomc_momentum > 0))

        # 负向脉冲 (看空美债 TLT):
        # 触发点 A: 央行发生极端的鹰派转向突变
        # 触发点 B: 在央行总体偏鹰的背景下，市场从极度自满中苏醒，不确定性开始飙升
        short_cond = fomc_hawkish_shock | (usepu_bottom_breakout & (fomc_momentum < 0))

        # 信号赋值
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        # 清除数据毛刺导致的同时触发冲突(理论上极罕见)
        conflict = long_cond & short_cond
        signal[conflict] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"