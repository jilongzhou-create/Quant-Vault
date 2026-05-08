import numpy as np
import pandas as pd

class UnstructuredFomcPivotFactor:
    """FOMC政策转向反转脉冲因子 (unstructured/NLP Sentiment)

    逻辑: 捕捉美联储政策周期的跨轴突变(Policy Pivot Shock)。当FOMC情绪得分的5日边际变化出现极端跳跃(Z-Score > 2.5)且跨越零轴时，意味着前期紧缩/宽松政策的动能彻底衰竭并出现预期逆转。信号将在突变发生及随后的极短几天内持续产生单向脉冲，之后迅速休眠。
    数据: fomc_sentiment (NLP 鹰鸽情绪得分)
    触发: 5日变化量的 252日滚动 Z-Score 绝对值 > 2.5，结合情绪绝对得分从负转正(鹰衰竭转鸽 -> 看多美债)或从正转负(鸽衰竭转鹰 -> 看空美债)。
    输出: 仅在反转确认的极短短期窗口内输出 +/- 1.0 的脉冲信号，常态休眠为 0.0。
    """

    def __init__(self, window: int = 252, z_threshold: float = 2.5, pulse_days: int = 5):
        self.name = 'unstructured_fomc_pivot_shock'
        self.window = window
        self.z_threshold = z_threshold
        self.pulse_days = pulse_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化信号，铁律1: 零值休眠
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖列缺失
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # 阶梯状低频数据前向填充，确保能平稳计算差分
        sentiment = data['fomc_sentiment'].ffill()
        
        # 铁律3: 边际变化，提取短期动量跳跃，捕捉会议声明发布当天的瞬时变差
        delta = sentiment.diff(self.pulse_days)
        
        # 计算一年期滚动 Z-Score，识别是否属于极端脉冲事件
        roll_mean = delta.rolling(window=self.window, min_periods=21).mean()
        roll_std = delta.rolling(window=self.window, min_periods=21).std()
        
        # 防止绝大多数无会议日 delta 均为 0 导致的除 0 错误
        roll_std = roll_std.replace(0.0, np.nan).fillna(0.0001)
        z_score = (delta - roll_mean) / roll_std
        
        # 铁律2: 二阶导数与防接飞刀，必须验证前期趋势耗尽且发生越轴反转
        prev_sentiment = sentiment.shift(self.pulse_days)
        curr_sentiment = sentiment
        
        # 多头脉冲：极其强烈的偏鸽跳跃 + 前期属于紧缩态(鹰派/负值)彻底衰竭并翻正
        cond_long = (z_score > self.z_threshold) & (prev_sentiment < 0.0) & (curr_sentiment > 0.0)
        
        # 空头脉冲：极其强烈的偏鹰跳跃 + 前期属于宽松态(鸽派/正值)彻底衰竭并翻负
        cond_short = (z_score < -self.z_threshold) & (prev_sentiment > 0.0) & (curr_sentiment < 0.0)
        
        # 赋值脉冲信号
        signal.loc[cond_long] = 1.0
        signal.loc[cond_short] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold}, pulse_days={self.pulse_days})"