import numpy as np
import pandas as pd

class UnstructuredFomcSentimentCapitulationFactor:
    """因子名称 (microstructure/unstructured)

    逻辑: 央行预期情绪是驱动美债的绝对锚。当长达数月的极端鹰派（或鸽派）预期将市场微观预期的结构推至极度拥挤后，
          一旦 FOMC 声明中出现超预期的边际情绪跳跃（投降式逆转），这种原趋势预期的崩塌会引发巨大的空头平仓或多头踩踏，产生爆炸性的单日强脉冲。
          本因子捕捉阶梯状 NLP 情绪得分的二阶导数跳跃，常态绝对休眠，仅在预期逆转跳跃当日实施狙击。
    数据: fomc_sentiment (基于 LLM 的央行文本非结构化情感得分)
    触发: 
          条件1 (极值): 过去 60 日的长期情绪均值处于绝对单边极值压制 (< -0.2 为长期鹰，> 0.2 为长期鸽)
          条件2 (衰竭与逆转): 当日 fomc_sentiment.diff() 出现强力反转 (绝对变动 > 0.1 且 252 日 Z-Score 的极值突破 2.5)
    输出: +1.0 (长期鹰压制下的鸽派投降逆转，抄底脉冲), -1.0 (长期鸽狂欢下的鹰派突变，看空脉冲), 否则 0.0
    """

    def __init__(self):
        self.name = 'fomc_sentiment_capitulation_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化常态 0.0 的狙击手休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        # 缺失保护
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # T+1 生效的情绪数据前向填充，构建稳定的阶梯数据
        fomc = data['fomc_sentiment'].ffill().fillna(0.0)
        
        # 铁律3: 阶梯数据的边际预期变化必须使用 .diff()
        # 由于数据长期水平不变，delta 仅在新的且意外的声明发布 T+1 当天非零
        delta = fomc.diff().fillna(0.0)
        
        # 衡量边际跳跃的波动程度
        delta_std = delta.rolling(window=252, min_periods=21).std()
        
        # 防止除零及全 0 环境导致的无效 Z-score，0.01 保护最小波动分母
        delta_std = delta_std.replace(0.0, np.nan).ffill().fillna(0.01)
        
        # 计算动量变化的 Z-Score
        delta_z = delta / delta_std
        
        # 铁律2: 前置极端状态判定。计算事件发生前 (shift) 的 60 日长周期情绪积淀
        # 必须 shift(1) 防止当天的新跳跃影响过去的背景判定
        prev_long_term_sentiment = fomc.shift(1).rolling(window=60, min_periods=21).mean().fillna(0.0)
        
        # 脉冲触发逻辑
        # 条件组合 A - 鸽派投降式逆转脉冲 (极值鹰转鸽 -> 抄底看多 TLT):
        # 1. 长期受鹰派极度压制 (均值 < -0.2)
        # 2. 边际变化剧烈反向，呈显著鸽派方向 (跳跃绝对值 > 0.1 且 Z-score > 2.5)
        bull_pulse = (prev_long_term_sentiment < -0.2) & (delta > 0.1) & (delta_z > 2.5)
        
        # 条件组合 B - 鹰派投降式逆转脉冲 (极值鸽转鹰 -> 阻击看空 TLT):
        # 1. 长期受鸽派放水狂欢 (均值 > 0.2)
        # 2. 边际变化剧烈反向，呈显著鹰派紧缩信号 (跳跃绝对值 < -0.1 且 Z-score < -2.5)
        bear_pulse = (prev_long_term_sentiment > 0.2) & (delta < -0.1) & (delta_z < -2.5)
        
        # 严格赋予极高爆发的单日脉冲信号
        signal.loc[bull_pulse] = 1.0
        signal.loc[bear_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"