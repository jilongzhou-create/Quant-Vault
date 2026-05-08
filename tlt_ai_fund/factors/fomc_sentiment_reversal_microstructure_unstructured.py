import numpy as np
import pandas as pd

class FomcSentimentReversalFactor:
    """FOMC情绪预期跳跃与反转因子 (microstructure/unstructured)

    逻辑: 将 NLP 提取的 FOMC 声明鹰鸽情绪得分转化为极短期的脉冲信号。
          此类低频阶梯状数据绝对水位无预测力，必须捕捉预期改变的瞬间。
          当情绪得分发生 2.5个标准差以上的极端反向跳跃 (例如从鹰派突然变鸽，且变动幅度极大)，
          才标志着极度恐慌的衰竭和政策预期的实质性扭转，输出脉冲信号抄底/做空美债。
    数据: fomc_sentiment
    触发: fomc_sentiment 的 5日变化量 Z-Score > 2.5 (预期跳跃) + 且伴随状态跨越零轴的衰竭反转 (前序状态与跳跃方向相反)。
    输出: +1.0 (鸽派强烈反转, 看多美债), -1.0 (鹰派强烈反转, 看空美债)。非触发常态严格休眠返回 0.0。
    """

    def __init__(self, window=252, diff_days=5, z_threshold=2.5, pulse_hold_days=5):
        self.name = 'fomc_sentiment_reversal'
        self.window = window
        self.diff_days = diff_days
        self.z_threshold = z_threshold
        self.pulse_hold_days = pulse_hold_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 非触发日信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # T+1 生效的阶梯状数据，向前填充处理 NaN
        sent = data['fomc_sentiment'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)，计算 5 日动量变化
        mom = sent.diff(self.diff_days)
        
        # 计算动量的滚动统计量，以评估预期突变的极端程度
        # 为了避免全 0 期间方差为 0，添加 1e-5 极小值保护
        roll_mean = mom.rolling(self.window).mean()
        roll_std = mom.rolling(self.window).std().replace(0.0, 1e-5)
        z_mom = (mom - roll_mean) / roll_std
        
        # 前序预期的绝对状态 (用于判断衰竭)
        prev_sent = sent.shift(self.diff_days)
        
        # 铁律2: 二阶导数 (极值突变 + 衰竭反转)
        # 看多脉冲：5日鸽派突跳极为罕见 (Z > 2.5)，且预期是从鹰派(负值)区域开始向鸽派逆转
        bull_pulse = (z_mom > self.z_threshold) & (prev_sent < 0.0)
        
        # 看空脉冲：5日鹰派突跳极为罕见 (Z < -2.5)，且预期是从鸽派(正值)区域开始向鹰派逆转
        bear_pulse = (z_mom < -self.z_threshold) & (prev_sent > 0.0)
        
        # 赋值触发日的脉冲信号
        signal.loc[bull_pulse] = 1.0
        signal.loc[bear_pulse] = -1.0
        
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 为使其成为极短期的有效可交易脉冲并达到目标的 Trigger Rate，向后延续维持数天
        # 其余时间信号严格通过 fillna(0.0) 保持休眠
        signal = signal.replace(0.0, np.nan).ffill(limit=self.pulse_hold_days - 1).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, diff_days={self.diff_days}, z_threshold={self.z_threshold}, pulse_hold_days={self.pulse_hold_days})"