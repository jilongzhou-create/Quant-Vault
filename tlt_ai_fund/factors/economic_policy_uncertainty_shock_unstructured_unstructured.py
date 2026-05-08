import numpy as np
import pandas as pd

class EconomicPolicyUncertaintyShockFactor:
    """经济政策不确定性脉冲 (unstructured/unstructured)

    逻辑: 经济政策不确定性(EPU)在短期内剧烈飙升通常暗示重大宏观风险爆发，这会驱动避险资金流入美债(TLT)。为了避免在不确定性发酵主升浪中“接飞刀”，必须等待EPU边际激增达到极值且其势头开始衰竭时，才触发确定的看多脉冲。反之，不确定性断崖式下跌暗示市场过度乐观，抛售美债，需等下杀衰竭后做空。
    数据: usepuindxd (经济政策不确定性指数，基于新闻和文本的非结构化日频指标)
    触发: 10日边际变化量的 252日 Z-Score > 2.5，且当日绝对量回落 (diff(1) < 0) 并跌破3日均值 -> 触发 +1.0; Z-Score < -2.5 且动量止跌 -> 触发 -1.0。
    输出: [-1.0, 1.0] 的狙击手级脉冲信号。
    """

    def __init__(self):
        self.name = 'epu_shock_pulse_unstructured'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 必须处理数据缺失的情况
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        epu = data['usepuindxd'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 计算 10 日边际变化，捕捉不确定性的短期急剧累积或消散突变
        epu_diff_10 = epu.diff(10)
        
        # 计算 252 日(约1年)滚动 Z-Score
        roll_mean = epu_diff_10.rolling(window=252, min_periods=126).mean()
        roll_std = epu_diff_10.rolling(window=252, min_periods=126).std()
        
        # 避免除以零导致无限大
        zscore = (epu_diff_10 - roll_mean) / roll_std.replace(0, np.nan)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 衰竭条件：当日指标绝对值开始反向运动，并且变化量本身的动量跌破3日均线
        daily_diff = epu.diff(1)
        epu_diff_10_ma3 = epu_diff_10.rolling(3).mean()
        
        momentum_exhaustion_up = (daily_diff < 0) & (epu_diff_10 < epu_diff_10_ma3)
        momentum_exhaustion_down = (daily_diff > 0) & (epu_diff_10 > epu_diff_10_ma3)
        
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 初始信号全为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 触发看多脉冲: 不确定性极度飙升 (Z > 2.5) 后见顶回落 (避险情绪确认落地)
        long_condition = (zscore > 2.5) & momentum_exhaustion_up
        
        # 触发看空脉冲: 不确定性极度骤降 (Z < -2.5) 后企稳 (市场极度乐观确认，抛售避险资产)
        short_condition = (zscore < -2.5) & momentum_exhaustion_down
        
        # 赋值极端事件的脉冲信号
        signal.loc[long_condition] = 1.0
        signal.loc[short_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"