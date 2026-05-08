import numpy as np
import pandas as pd

class FomcPivotShockFactor:
    """FOMC预期反转突变 (政策预期突变/非结构化数据)

    逻辑: 捕捉美联储政策预期的极端反转跳跃。FOMC声明情绪得分为低频阶梯状数据，每次会议T+1更新。当情绪得分发生超预期的极端跳跃（边际差分的极值）且打破前期的政策倾向锚定（如前期偏鹰，旧逻辑衰竭并向上逆转为鸽派）时，产生捕捉定价时差的狙击手级别脉冲。
    数据: fomc_sentiment
    触发: fomc_sentiment 5日变化量的 252日 Z-Score 极值 (>2.5或<-2.5) + 前期得分处于反向区间 (衰竭逆转) + 短期阶跃动作确认。
    输出: +1.0 (鹰转鸽脉冲看多美债), -1.0 (鸽转鹰脉冲看空美债)
    """

    def __init__(self):
        self.name = 'fomc_pivot_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1：零值休眠，初始所有非触发日信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # 填充非会议日的阶梯平段
        fomc = data['fomc_sentiment'].ffill()
        
        # 铁律3：边际变化 (Marginal Change Only)
        # 绝对禁止直接输出情绪得分绝对值，使用 5 个交易日的差分将低频阶梯转化为爆发脉冲动量
        fomc_diff5 = fomc.diff(5)
        
        # 计算该边际变化的极端程度 (动态统计锚)
        rolling_mean = fomc_diff5.rolling(window=252, min_periods=63).mean()
        rolling_std = fomc_diff5.rolling(window=252, min_periods=63).std()
        
        # 1e-6 用于防止除零，Z-Score 衡量边际跳变是否超预期
        z_score = (fomc_diff5 - rolling_mean) / (rolling_std + 1e-6)
        
        # 铁律2：二阶导数衰竭 (Anti-Catch-Falling-Knife)
        # 孤立的剧烈跳跃仍不充分，必须证明“原有的政策预期已经发生衰竭并反转”
        # 获取发生跳跃前的历史情绪倾向
        prev_fomc = fomc.shift(5)
        
        # 鸽派突变脉冲: 突发鸽派大跳跃 (Z > 2.5) + 且打破了原先偏鹰派的常态 (< 0.0) -> 原有紧缩动能衰竭
        is_bull_shock = (z_score > 2.5) & (prev_fomc < 0.0) & (fomc > prev_fomc)
        
        # 鹰派突变脉冲: 突发鹰派大跳跃 (Z < -2.5) + 且打破了原先偏鸽派的常态 (> 0.0) -> 原有宽松动能衰竭
        is_bear_shock = (z_score < -2.5) & (prev_fomc > 0.0) & (fomc < prev_fomc)
        
        # 为确保信号不会在差分高位期无意义顺延，使用 diff(1) 锚定真实的物理阶跃瞬间
        # 只有在过去 5 天内真实发生过离散会议跳跃，才允许信号释放
        is_event_triggered = fomc.diff(1).abs().rolling(5).max() > 0
        
        # 脉冲点爆破
        signal[is_bull_shock & is_event_triggered] = 1.0
        signal[is_bear_shock & is_event_triggered] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"