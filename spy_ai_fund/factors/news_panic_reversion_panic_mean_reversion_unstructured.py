import numpy as np
import pandas as pd

class FomcSentimentPulseFactor:
    """FOMC Sentiment Pivot Pulse (Unstructured)

    逻辑: 捕捉FOMC声明中鹰鸽态度的边际突变。美股对货币政策预期的二阶导数极度敏感。极度鹰派后的边际缓和代表'恐慌衰竭'(看多); 突发鹰派代表'轻微恐慌/趋势恶化'(看空)。
    数据: [fomc_sentiment]
    输出: 鸽派突变或鹰派衰竭输出+1.0; 鹰派突变输出-1.0。
    触发条件: FOMC情绪边际变化发生当日及随后2天。每年约8次会议，预期Trigger Rate控制在 5%-15% 区间。
    """

    def __init__(self):
        self.name = 'fomc_sentiment_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 必须处理数据缺失的情况
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        # FOMC情绪得分：1.0=极度鸽派，-1.0=极度鹰派
        # 会议外日期前向填充，边际突变(diff)只在会议日发生
        fomc = data['fomc_sentiment'].ffill()
        fomc_diff = fomc.diff()
        
        # 1. 鸽派突变 (Dovish Shock): 边际向鸽派大幅移动 (利好美股)
        dovish_shock = fomc_diff > 0.15
        
        # 2. 鹰派突变 (Hawkish Shock): 边际向鹰派大幅移动 -> 导致风险偏好受挫恶化 (利空美股)
        hawkish_shock = fomc_diff < -0.15
        
        # 3. 二阶导数/恐慌衰竭 (Hawkish Exhaustion): 之前处于极度鹰派状态(<-0.4)，现在边际开始缓和(>0.05) -> 紧缩预期见顶回落，绝佳抄底买点 (+1.0)
        hawkish_exhaustion = (fomc.shift(1) <= -0.4) & (fomc_diff > 0.05)
        
        # 聚合触发器
        trigger_buy = dovish_shock | hawkish_exhaustion
        trigger_sell = hawkish_shock & (~trigger_buy)
        
        # 零值休眠铁律: 生成 3 天的有效脉冲窗口，衰减至0.0
        # 8次会议 * 3天 = 全年24天激活，完美卡位 5%-15% 目标 Trigger Rate
        buy_pulse = trigger_buy.astype(float).rolling(window=3, min_periods=1).max()
        sell_pulse = trigger_sell.astype(float).rolling(window=3, min_periods=1).max()
        
        # 组装最终信号 (默认全0.0)
        signal = pd.Series(0.0, index=data.index)
        
        # 负向看空信号写入
        signal[sell_pulse > 0] = -1.0
        
        # 正向看多信号写入 (极端底部反转享有最高优先级)
        signal[buy_pulse > 0] = 1.0 
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"