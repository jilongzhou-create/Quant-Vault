import numpy as np
import pandas as pd

class FomcHawkishPanicReversionFactor:
    """FOMC情绪恐慌均值回归因子 (panic_mean_reversion/unstructured)

    逻辑: 捕捉美联储预期从"鹰派恐慌"极值区域边际转向鸽派的瞬间。当FOMC声明情绪此前处于负面(鹰派预期)且最新会议的边际变化显著转暖(情绪突变)时，意味着恐慌衰竭，构成买点；反向则构成卖点。信号在触发后持续5个交易日形成短期资金配置窗口。
    数据: fomc_sentiment
    输出: +1.0 看多美股(鹰派恐慌衰竭), -1.0 看空美股(鸽派狂热衰竭)
    触发条件: 情绪前值偏鹰(<0)且边际转鸽(>0.15)，或前值偏鸽(>0)且边际转鹰(<-0.15)，延展4天，预期Trigger Rate在5%-10%区间
    """

    def __init__(self):
        self.name = 'fomc_hawkish_panic_reversion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查数据列是否存在
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        # 获取 T+1 生效的低频数据并进行前向填充
        sentiment = data['fomc_sentiment'].ffill()
        
        # 计算前值和边际变化(捕捉离散跳跃的瞬间，绝对遵守边际变化铁律)
        prev_sentiment = sentiment.shift(1)
        sentiment_diff = sentiment.diff()
        
        # 建立极值与衰竭的二阶导逻辑
        # 1. 鹰派恐慌见顶回落 (买入信号)
        # 条件：之前市场处于偏鹰派恐慌中(< 0.0) 且 最新情绪边际大幅转鸽(> 0.15)
        # 或：之前极度鹰派(<= -0.1) 且 最新情绪直接反转为正(>= 0.0)
        bull_pulse = ((prev_sentiment < 0.0) & (sentiment_diff >= 0.15)) | \
                     ((prev_sentiment <= -0.1) & (sentiment >= 0.0))
                     
        # 2. 鸽派狂热见顶回落 (轻度防守信号)
        # 条件：之前市场处于鸽派狂热中(> 0.0) 且 最新情绪边际大幅转鹰(< -0.15)
        # 或：之前极度鸽派(>= 0.1) 且 最新情绪直接反转为负(<= 0.0)
        bear_pulse = ((prev_sentiment > 0.0) & (sentiment_diff <= -0.15)) | \
                     ((prev_sentiment >= 0.1) & (sentiment <= 0.0))
        
        # 脉冲信号初始化，默认全0休眠
        pulse_signal = pd.Series(0.0, index=data.index)
        pulse_signal[bull_pulse] = 1.0
        pulse_signal[bear_pulse] = -1.0
        
        # 【零值休眠与Trigger Rate铁律控制】
        # FOMC每年仅8次会议。为保证目标 Trigger Rate 落在 5%-15% 之间
        # 将触发当天的非零脉冲向后延展4个交易日(共计5天生命周期)
        signal = pulse_signal.replace(0.0, np.nan).ffill(limit=4).fillna(0.0)
        
        # 截断以确保信号极端稳健在[-1.0, 1.0]
        signal = signal.clip(-1.0, 1.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"