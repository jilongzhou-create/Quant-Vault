import numpy as np
import pandas as pd

class FomcNlpSentimentReversalFactor:
    """Unstructured NLP FOMC Sentiment Pivot Factor

    逻辑: 捕捉美联储FOMC声明文本情绪的极端突变与反转。当NLP鹰鸽情绪的5日边际变化出现极大极值(Z-Score>2.5)，且情绪绝对得分跨越零轴实现实质性反转(鹰转鸽或鸽转鹰)时，代表政策预期发生未被市场完全Price-in的根本性跳跃。此时输出脉冲信号，并维持短短几日以捕捉情绪突变的冲击波。
    数据: fomc_sentiment (NLP情绪得分, 1.0=极鸽, -1.0=极鹰)
    触发: 边际变化极端化 (fomc_sentiment.diff(5)的252日Z-Score > 2.5) + 衰竭与反转 (情绪得分跨越0轴)
    输出: 脉冲型信号。鹰转鸽突变触发+1.0(看多美债)并持续7个交易日，鸽转鹰突变触发-1.0(看空美债)并持续7个交易日，常态下严格保持0.0。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_sentiment_reversal_pulse'
        self.window = 252           # 1个自然年的滚动窗口，评估突变级别
        self.diff_days = 5          # 5日变化量，充分吸收T+1生效的会议情绪跳跃
        self.z_thresh = 2.5         # 极端脉冲阈值，过滤常规的鸽派/鹰派微调
        self.min_change = 0.15      # 绝对变化量的最小经济学阈值，防止在历史极低波动期因Z-Score失真而误触
        self.pulse_hold = 7         # 脉冲维持时间(极短几天)，确保Trigger Rate在5%-15%区间并捕捉完事件冲击浪

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始及非触发日严格为 0.0
        signal = pd.Series(0.0, index=data.index)

        if 'fomc_sentiment' not in data.columns:
            return signal

        # 处理低频阶梯数据的填充
        sent = data['fomc_sentiment'].ffill()

        # 铁律3: 边际变化 (绝对禁止直接使用 fomc_sentiment 的绝对水位)
        sent_diff = sent.diff(self.diff_days)

        # 滚动均值与标准差计算 Z-Score
        roll_mean = sent_diff.rolling(window=self.window, min_periods=self.window//2).mean()
        roll_std = sent_diff.rolling(window=self.window, min_periods=self.window//2).std().replace(0.0, 1e-5)
        
        z_score = (sent_diff - roll_mean) / roll_std

        # 铁律2: 二阶导数 (极端位 + 衰竭/反转确认)
        # 条件1: 鸽派突变 -> Z-Score > 2.5 且绝对变化量足够大
        # 条件2: 衰竭与反转 -> 当前情绪越过0轴进入鸽派(sent>0)，且5天前仍处于鹰派或中性(sent<=0)
        dovish_trigger = (
            (z_score > self.z_thresh) & 
            (sent_diff > self.min_change) & 
            (sent > 0.0) & 
            (sent.shift(self.diff_days) <= 0.0)
        )

        # 条件1: 鹰派突变 -> Z-Score < -2.5 且绝对变化量足够大
        # 条件2: 衰竭与反转 -> 当前情绪越过0轴进入鹰派(sent<0)，且5天前仍处于鸽派或中性(sent>=0)
        hawkish_trigger = (
            (z_score < -self.z_thresh) & 
            (sent_diff < -self.min_change) & 
            (sent < 0.0) & 
            (sent.shift(self.diff_days) >= 0.0)
        )

        # 构建狙击手脉冲原始事件
        raw_events = pd.Series(np.nan, index=data.index)
        raw_events.loc[dovish_trigger] = 1.0
        raw_events.loc[hawkish_trigger] = -1.0

        # 在触发后的极短几天内维持信号，确保能够捕捉到事件完整的Price-in过程，同时符合 5%-15% 的目标触发率
        signal_filled = raw_events.ffill(limit=self.pulse_hold)

        # 将没有被事件脉冲覆盖的日期重新归零
        signal = signal_filled.fillna(0.0)
        signal.name = self.name

        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, diff_days={self.diff_days}, z_thresh={self.z_thresh})"