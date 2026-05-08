import numpy as np
import pandas as pd

class FomcSentimentInflectionFactor:
    """FOMC情绪边际反转脉冲因子 (unstructured/unstructured)

    逻辑: 捕捉美联储预期管理在极端状态下的边际衰竭与反转。因为FOMC情绪得分为低频阶梯状数据，连续输出绝对水位会导致在主跌浪中接飞刀。本因子严格遵守边际变化与二阶导数铁律，计算情绪得分的5日动量变化。仅当变化量超过2.5倍标准差（超预期极值跳跃），且情绪绝对值发生跨越零轴的根本性反转时，才触发狙击手级别的短周期脉冲信号。
    数据: fomc_sentiment
    触发: 5日变化量Z-Score > 2.5 且 情绪从负转正 -> 看多(鹰派极值衰竭转鸽)；Z-Score < -2.5 且 情绪从正转负 -> 看空(鸽派极值衰竭转鹰)
    输出: 严格脉冲型信号 [-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'fomc_sentiment_inflection_unstructured'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 零值休眠铁律：初始赋值全为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 防错处理
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # 1. 基础数据处理，前向填充以处理非会议日的缺失值
        sentiment = data['fomc_sentiment'].ffill()
        
        # 2. 边际变化铁律：计算5日变化量 (捕捉政策转向引发的极短几天定价期)
        change_5d = sentiment.diff(5)
        
        # 3. 零值休眠与极值铁律：计算252个交易日(约1年)滚动Z-Score，仅捕捉具有统计学意义的突变
        mean_252 = change_5d.rolling(window=252, min_periods=21).mean()
        std_252 = change_5d.rolling(window=252, min_periods=21).std()
        # 替换 std 为 0 的情况为 NaN，防止除以 0 导致错误极值
        z_score = (change_5d - mean_252) / std_252.replace(0, np.nan)
        
        # 4. 二阶导数/衰竭防飞刀铁律：禁止仅凭变化极值入场，必须伴随原有状态的彻底反转
        prev_sentiment = sentiment.shift(5)
        curr_sentiment = sentiment
        
        # 看多脉冲：预期向鸽派极速突变 (超2.5σ) + 鹰派彻底衰竭 (过去5天曾小于0，现在>=0)
        long_pulse = (z_score > 2.5) & (prev_sentiment < 0.0) & (curr_sentiment >= 0.0)
        
        # 看空脉冲：预期向鹰派极速突变 (低于-2.5σ) + 鸽派彻底衰竭 (过去5天曾大于0，现在<=0)
        short_pulse = (z_score < -2.5) & (prev_sentiment > 0.0) & (curr_sentiment <= 0.0)
        
        # 5. 脉冲输出：由于使用的是5日差分，信号将在会议突变后的5个交易日内维持脉冲输出，符合 Trigger Rate 约束
        signal[long_pulse] = 1.0
        signal[short_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"