import numpy as np
import pandas as pd

class FomcSentimentJumpPulseFactor:
    """FOMC情绪跃升脉冲因子 (policy_pivot/unstructured)

    逻辑: 捕捉美联储声明在鹰鸽情绪上发生边际突变的极短窗口期。绝对不看情绪绝对值，只捕捉预期发生突发改变的瞬间（情绪分出现跳跃）。鸽派突变刺激风险偏好，鹰派突变打压估值。
    数据: [fomc_sentiment]
    输出: 鸽派跃升触发+1.0脉冲，鹰派急挫触发-1.0脉冲，平息期恢复为0.0
    触发条件: 情绪得分日环比变化幅度 > 0.10（约代表10%的预期突转），市场利用随后4天时间消化该突变，预期Trigger Rate在5%-10%之间。
    """

    def __init__(self, jump_threshold=0.10, pulse_window=4):
        self.name = 'fomc_sentiment_jump_pulse'
        self.jump_threshold = jump_threshold
        self.pulse_window = pulse_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        # 前向填充，维持阶梯状特征
        sentiment = data['fomc_sentiment'].ffill()
        
        # 边际变化铁律: FOMC是低频阶梯数据，必须使用.diff()计算动量跳跃
        sentiment_diff = sentiment.diff()
        
        # 捕捉鸽派跳跃(大幅转正向)与鹰派急挫(大幅转负向)
        dovish_jump = (sentiment_diff > self.jump_threshold).astype(int)
        hawkish_drop = (sentiment_diff < -self.jump_threshold).astype(int)
        
        # 零值休眠铁律: 仅在突变发生日及随后极短的消化窗口期内（如4天）输出信号
        dovish_pulse = dovish_jump.rolling(window=self.pulse_window, min_periods=1).max()
        hawkish_pulse = hawkish_drop.rolling(window=self.pulse_window, min_periods=1).max()
        
        # 鸽派买入脉冲 (+1.0)，鹰派卖出脉冲 (-1.0)
        signal = dovish_pulse - hawkish_pulse
        
        # 处理异常并规范输出范围
        signal = signal.fillna(0.0).clip(-1.0, 1.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(jump_threshold={self.jump_threshold}, pulse_window={self.pulse_window})"