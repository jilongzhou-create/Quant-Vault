import numpy as np
import pandas as pd

class UnstructuredEpuVolSentimentReversalFactor:
    """Unstructured EPU Volatility & Sentiment Reversal Factor (volatility/unstructured)

    逻辑: 结合基于新闻的经济政策不确定性(EPU)与基于NLP的FOMC文本情绪得分，提取非结构化数据的“叙事波动率”。当EPU的短期波动率达到历史极端高位(Z-Score>2.5)且开始明确衰竭回落时，标志着宏观恐慌叙事开始瓦解。此时若近期FOMC情绪处于极端鹰派拥挤，说明加息恐慌见顶反转，脉冲做多美债(TLT)；若处于极端鸽派拥挤，说明衰退放水恐慌见顶，经济基本面复苏，脉冲做空美债。严格遵守狙击手零值休眠铁律。
    数据: usepuindxd (经济政策不确定性指数), fomc_sentiment (FOMC央行声明情绪得分)
    触发: EPU的21日波动率 252日 Z-Score > 2.5 且回落 (diff < 0) + 60日情绪均值极值
    输出: 脉冲信号，做多(+1.0) / 做空(-1.0) / 常态(0.0)
    """

    def __init__(self):
        self.name = 'unstructured_epu_vol_sentiment_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态下必须返回0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需卫星数据是否存在 (禁止使用 CoreAnchor 数据)
        if 'usepuindxd' not in data.columns or 'fomc_sentiment' not in data.columns:
            return signal
            
        # 非结构化数据通常存在低频/阶梯跳跃特征，进行前向填充
        epu = data['usepuindxd'].ffill()
        sentiment = data['fomc_sentiment'].ffill()
        
        # --------------------------------------------------------
        # 核心逻辑一: 提取非结构化数据的异常波动率 (Unstructured Volatility Extreme)
        # 使用 21日标准差衡量新闻层面政策不确定性的近期“狂飙”程度
        epu_vol = epu.rolling(window=21).std()
        
        # 计算该波动率的年度 252日 Z-Score 以识别宏观极端尾部事件
        epu_vol_mean = epu_vol.rolling(window=252).mean()
        epu_vol_std = epu_vol.rolling(window=252).std().replace(0, np.nan)
        epu_vol_zscore = (epu_vol - epu_vol_mean) / epu_vol_std
        
        # --------------------------------------------------------
        # 核心逻辑二: 二阶导数反接飞刀 (Anti-Catch-Falling-Knife)
        # 条件1: 叙事波动率处于极端高位
        vol_extreme = epu_vol_zscore > 2.5
        
        # 铁律2: 绝对禁止直接极值买入，必须叠加动量衰竭
        # 条件2: 波动率自身开始回落，且EPU绝对水位也低于3日均线，确认恐慌动量消退
        vol_exhaustion = (epu_vol.diff() < 0) & (epu < epu.rolling(window=3).mean())
        
        # --------------------------------------------------------
        # 核心逻辑三: NLP边际变化与拥挤确认 (Sentiment Crowding)
        # 铁律3: 禁止直接使用绝对值，通过 60日均线提取低频数据的宏观叙事“底色”
        sentiment_trend = sentiment.rolling(window=60).mean()
        
        # fomc_sentiment: 1.0=极度鸽派, -1.0=极度鹰派
        hawkish_crowding = sentiment_trend < -0.25
        dovish_crowding = sentiment_trend > 0.25
        
        # --------------------------------------------------------
        # 触发器生成
        # 鹰派拥挤下的恐慌瓦解 -> 加息预期到头，避险资产与降息预期共振 -> 脉冲做多美债
        raw_bull = vol_extreme & vol_exhaustion & hawkish_crowding
        
        # 鸽派拥挤下的恐慌瓦解 -> 衰退与放水预期消退，经济绿芽复苏 -> 脉冲做空美债
        raw_bear = vol_extreme & vol_exhaustion & dovish_crowding
        
        # 为确保目标 Trigger Rate 落在 5%-15% 的黄金区间，将罕见的极值突破脉冲向后延展3日
        # 且使用 min_periods=1 和 astype(float) 防止早期的 NaN 感染
        pulse_bull = raw_bull.astype(float).rolling(window=3, min_periods=1).max() > 0
        pulse_bear = raw_bear.astype(float).rolling(window=3, min_periods=1).max() > 0
        
        # 信号赋值
        signal.loc[pulse_bull] = 1.0
        signal.loc[pulse_bear & ~pulse_bull] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"