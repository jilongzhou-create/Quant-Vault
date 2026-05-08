import numpy as np
import pandas as pd

class FomcSentimentPivotPulseFactor:
    """FOMC情绪反转脉冲 (unstructured/unstructured)

    逻辑: 捕捉美联储从鹰派到鸽派(或反之)的超预期突然转向。直接使用绝对情绪分数会导致持续看多/看空(接飞刀)，必须通过边际变化与象限反转来捕捉定价重估的瞬间脉冲。
    数据: fomc_sentiment
    触发: 5日情绪动量 Z-Score > 2.5 (预期跳跃突变极值) 且 从负转正/从正转负 (先前状态衰竭并发生性质反转)
    输出: +1.0 表示向鸽派突变看多美债，-1.0 表示向鹰派突变看空美债，常态为 0.0。
    """

    def __init__(self, window_diff=5, window_z=252, z_threshold=2.5):
        self.name = 'fomc_sentiment_pivot_pulse'
        self.window_diff = window_diff
        self.window_z = window_z
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        # 获取基础数据并进行前向填充以应对阶梯状非会议日
        sentiment = data['fomc_sentiment'].ffill()
        
        # 1. 边际变化铁律: 绝对禁止使用绝对值，计算低频数据的跳跃变化量 (动量)
        delta = sentiment.diff(self.window_diff)
        
        # 2. 计算极值: 历史滚动 Z-Score 
        # 为了避免非会议日产生大量0导致标准差为0，加上 1e-6 避免除零
        roll_mean = delta.rolling(self.window_z).mean()
        roll_std = delta.rolling(self.window_z).std() + 1e-6
        zscore = (delta - roll_mean) / roll_std
        
        # 3. 二阶导数/反转铁律: 判断前期状态是否发生衰竭和性质跨越
        prev_sentiment = sentiment.shift(self.window_diff)
        curr_sentiment = sentiment
        
        # 鸽派突变看多条件: 情绪正向剧烈跳变 + 前期属于鹰派(<0) + 当期已跨越零轴反转为鸽派(>0)
        bull_condition = (
            (zscore > self.z_threshold) & 
            (prev_sentiment < 0) & 
            (curr_sentiment > 0)
        )
        
        # 鹰派突变看空条件: 情绪负向剧烈跳变 + 前期属于鸽派(>0) + 当期已跨越零轴反转为鹰派(<0)
        bear_condition = (
            (zscore < -self.z_threshold) & 
            (prev_sentiment > 0) & 
            (curr_sentiment < 0)
        )
        
        # 4. 零值休眠铁律: 生成狙击手级脉冲信号
        signal = pd.Series(0.0, index=data.index)
        signal.loc[bull_condition] = 1.0
        signal.loc[bear_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window_diff={self.window_diff}, window_z={self.window_z}, z_threshold={self.z_threshold})"