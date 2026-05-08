import numpy as np
import pandas as pd

class PolicyUncertaintyReversalFactor:
    """政策不确定性恐慌衰竭与突变脉冲 (Microstructure / Unstructured)

    逻辑: 采用基于新闻文本挖掘的经济政策不确定性指数(USEPUINDXD)。当非结构化宏观恐慌飙升至极值时，会导致微观流动性干涸及资产错杀，此时必须遵守二阶导数铁律，绝对禁止在抛售浪潮中接飞刀，必须等待恐慌见顶且边际衰竭(环比回落)的瞬间，确认危机解除后抄底美债；反之，当市场极度自满(不确定性处于历史极低位)且突发不确定性跳升时，触发看空美债的脉冲。本因子平时严格零值休眠，仅在流动性情绪拐点输出狙击手级信号。
    数据: usepuindxd (每日美国经济政策不确定性指数，非结构化文本衍生数据)
    触发: 
      - 看多(+1.0): 过去5天不确定性 252日 Z-Score 曾 > 2.5(恐慌极值)，且当日值低于3日均值并环比下降(确认极值衰竭反转)。
      - 看空(-1.0): 过去5天不确定性 252日 Z-Score 曾 < -2.0(极度自满)，且当日值高于3日均值并环比上升(确认突变跳升)。
    输出: 脉冲型信号 [-1.0, 1.0]，常态严格返回 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_policy_uncertainty_reversal_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 初始信号必须为0.0 (零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        # 容错处理: 如果所需非结构化数据缺失，直接返回全0
        if 'usepuindxd' not in data.columns:
            signal.name = self.name
            return signal

        epu = data['usepuindxd'].ffill()

        # 数据量不足以计算长期Z-Score时，保持休眠
        if len(epu) < 252:
            signal.name = self.name
            return signal

        # 铁律2: 计算二阶导数的基础 - 衡量长期水位的极端程度
        roll_mean = epu.rolling(window=252, min_periods=126).mean()
        roll_std = epu.rolling(window=252, min_periods=126).std()
        z_score = (epu - roll_mean) / (roll_std + 1e-8)

        # 极值条件: 寻找过去5天内是否触及过极端恐慌高位或极度自满低位 (给市场足够筑顶/筑底时间)
        extreme_high_reached = z_score.rolling(window=5).max() > 2.5
        extreme_low_reached = z_score.rolling(window=5).min() < -2.0

        # 铁律2 & 3: 衰竭与边际变化条件 - 短期均线交叉与当日动量方向
        ma_3 = epu.rolling(window=3).mean()
        
        # 恐慌衰竭: 当日值跌破短期均线，且确认为下降趋势 (diff < 0)
        exhaustion_down = (epu < ma_3) & (epu.diff() < 0)
        
        # 自满突变: 当日值升破短期均线，且确认为上升趋势 (diff > 0)
        shock_up = (epu > ma_3) & (epu.diff() > 0)

        # 逻辑组合：极值区域 + 边际反转
        long_trigger = extreme_high_reached & exhaustion_down
        short_trigger = extreme_low_reached & shock_up

        # 进一步强化铁律1：只在拐点刚发生的"瞬间"触发1天脉冲，不连续持仓
        is_new_long = long_trigger & (~long_trigger.shift(1).fillna(False))
        is_new_short = short_trigger & (~short_trigger.shift(1).fillna(False))

        # 信号赋值
        signal[is_new_long] = 1.0
        signal[is_new_short] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"