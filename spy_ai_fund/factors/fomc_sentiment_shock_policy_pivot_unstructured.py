import numpy as np
import pandas as pd

class FomcSentimentShockFactor:
    """FOMC情绪突变脉冲因子 (policy_pivot/unstructured)

    逻辑: 捕捉美联储预期的剧烈变化。不关注FOMC声明情绪得分的绝对水平，而是关注其发生跳跃(Jump)的瞬间。
          当NLP情绪得分在两次会议之间出现超过25%的极性偏移，或完成跨越零轴的鹰鸽反转时，视为政策拐点。
    数据: [fomc_sentiment]
    输出: +1.0 (强烈鸽派转向，看多), -1.0 (强烈鹰派转向，看空), 常态返回 0.0
    触发条件: 仅在美联储声明情绪发生边际剧变的当日触发，并维持5个交易日的操作窗口，预期 Trigger Rate 约 5%-10%
    """

    def __init__(self, shock_threshold=0.25, turn_threshold=0.1, pulse_window=5):
        self.name = 'fomc_sentiment_shock'
        # 0.25 代表情绪得分发生显著的剧烈跳跃 (区间为-1到1，0.25具有明显的表态修改含义)
        self.shock_threshold = shock_threshold
        # 0.1 代表跨越零轴时，必须具备实质性的边际改变，过滤掉零点附近的微小噪音
        self.turn_threshold = turn_threshold
        # 脉冲维持的交易日数，给予建仓窗口
        self.pulse_window = pulse_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 如果数据缺失，休眠返回0.0
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        # 阶梯状的低频数据前向填充
        fomc = data['fomc_sentiment'].ffill()
        
        # 绝对遵守【边际变化铁律】，计算动量跳跃
        diff = fomc.diff(1).fillna(0.0)
        prev_fomc = fomc.shift(1).fillna(0.0)
        
        # 1. 鸽派突变: 得分单次正向跳升极大
        dovish_jump = diff >= self.shock_threshold
        # 2. 鸽派反转: 前次会议偏鹰或中立(<=0)，本次转鸽(>0)，且边际改变具有实质意义
        dovish_turn = (prev_fomc <= 0.0) & (fomc > 0.0) & (diff >= self.turn_threshold)
        
        # 3. 鹰派突变: 得分单次负向跳降极大
        hawkish_jump = diff <= -self.shock_threshold
        # 4. 鹰派反转: 前次会议偏鸽或中立(>=0)，本次转鹰(<0)，且边际改变具有实质意义
        hawkish_turn = (prev_fomc >= 0.0) & (fomc < 0.0) & (diff <= -self.turn_threshold)
        
        # 构建会议发布当日的狙击点脉冲
        raw_pulse = pd.Series(0.0, index=data.index)
        raw_pulse[dovish_jump | dovish_turn] = 1.0
        raw_pulse[hawkish_jump | hawkish_turn] = -1.0
        
        # 将瞬间的突变信号沿时间轴平移，形成一个持续 pulse_window 天的入场窗口脉冲
        signal = pd.Series(0.0, index=data.index)
        for i in range(self.pulse_window):
            shifted = raw_pulse.shift(i).fillna(0.0)
            signal[shifted == 1.0] = 1.0
            signal[shifted == -1.0] = -1.0
            
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(shock_threshold={self.shock_threshold}, pulse_window={self.pulse_window})"