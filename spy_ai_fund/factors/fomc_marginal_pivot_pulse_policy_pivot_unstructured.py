import numpy as np
import pandas as pd

class FomcMarginalPivotPulseFactor:
    """FOMC情绪边际转向脉冲因子 (policy_pivot/unstructured)

    逻辑: 捕捉美联储FOMC声明情绪的超预期突变(鸽派剧变看多, 鹰派剧变看空)。绝不使用绝对情绪得分，只关注阶梯状数据的边际跳跃，捕捉政策预期改变瞬间爆发的流动性冲量。
    数据: fomc_sentiment
    输出: +1.0 (鸽派突变看多), -1.0 (鹰派突变看空), 0.0 (常态休眠)
    触发条件: FOMC情绪得分单日边际跳变超过 0.15 (象征政策基调出现实质性跨档), 信号持续 5 个交易日(资金重新配置的一个交易周)。预期 Trigger Rate 约 8%-12%。
    """

    def __init__(self, jump_threshold: float = 0.15, pulse_window: int = 5):
        self.name = 'fomc_marginal_pivot_pulse'
        self.jump_threshold = jump_threshold  # 边际突变阈值
        self.pulse_window = pulse_window      # 脉冲延续窗口(美股消化政策转向的效率窗口)

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 缺少必要字段时返回全0
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        # 获取FOMC声明情绪得分 (非会议日为前向填充的平阶数据)
        sentiment = data['fomc_sentiment'].ffill()
        
        # 核心铁律: 绝对禁止使用绝对值，只计算动量边际变化
        # 在新的FOMC声明生效日，阶梯数据会发生跳跃
        marginal_change = sentiment.diff(1)
        
        # 寻找超出阈值的政策转向
        # 鸽派突变: 边际变化大于正向阈值
        dovish_jump = (marginal_change > self.jump_threshold).astype(int)
        # 鹰派突变: 边际变化小于负向阈值
        hawkish_jump = (marginal_change < -self.jump_threshold).astype(int)
        
        # 脉冲铁律: 将瞬间的转向信号延续到一个交易周内，以便捕捉完整的市场流动性冲量
        # 过去 pulse_window 天内发生过跳变则持续触发脉冲
        dovish_pulse = dovish_jump.rolling(window=self.pulse_window, min_periods=1).max()
        hawkish_pulse = hawkish_jump.rolling(window=self.pulse_window, min_periods=1).max()
        
        signal = pd.Series(0.0, index=data.index)
        
        # 鸽派转向看多，鹰派转向看空
        signal[dovish_pulse > 0] = 1.0
        signal[hawkish_pulse > 0] = -1.0
        
        # 防御性冲突处理 (现实中短期内极少出现双向极端跳变)
        conflict = (dovish_pulse > 0) & (hawkish_pulse > 0)
        signal[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(jump_threshold={self.jump_threshold}, pulse_window={self.pulse_window})"