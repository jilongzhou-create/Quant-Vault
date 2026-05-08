import numpy as np
import pandas as pd

class NlpSentimentPulseFactor:
    """NLP 非结构化情绪脉冲因子 (microstructure / unstructured)

    逻辑: 结合了两个高质量非结构化 NLP 文本数据。1. 政策不确定性恐慌衰竭：当新闻经济政策不确定性(EPU)飙升至极端恐慌水平(Z-Score>2.5)并开始回落时，标志流动性危机解除或极端避险情绪落地，美债迎来修复反弹。2. FOMC情绪边际突变：当FOMC声明的NLP鹰鸽得分出现预期外跳跃(Z-Score>2.5)且态度发生反转时，第一时间顺势押注美债。
    数据: usepuindxd (每日新闻政策不确定性), fomc_sentiment (FOMC文本鹰鸽情绪得分)
    触发: (EPU Z-Score > 2.5 且当天回落到3日均值以下) 或 (FOMC 情绪跳跃 Z-Score > 2.5 且反转)
    输出: +1.0 (恐慌衰竭或预期变鸽，看多美债), -1.0 (预期变鹰，看空美债)，脉冲持续3天。
    """

    def __init__(self):
        self.name = 'nlp_sentiment_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化常态零值，遵守“零值休眠”铁律
        signal = pd.Series(0.0, index=data.index)
        
        has_epu = 'usepuindxd' in data.columns
        has_fomc = 'fomc_sentiment' in data.columns
        
        if not has_epu and not has_fomc:
            return signal

        long_cond = pd.Series(False, index=data.index)
        short_cond = pd.Series(False, index=data.index)

        # 1. 每日政策不确定性 (EPU) 脉冲 - 遵守“二阶导数”铁律
        if has_epu:
            epu = data['usepuindxd'].ffill()
            epu_mean = epu.rolling(window=252, min_periods=126).mean()
            epu_std = epu.rolling(window=252, min_periods=126).std().replace(0.0, np.nan).bfill().fillna(0.001)
            epu_z = (epu - epu_mean) / epu_std
            
            # 条件1: 处于极端恐慌极值 (Z-Score > 2.5)
            # 条件2: 恐慌开始衰竭见顶 (当前值 < 过去3天均值)
            epu_exhaustion = epu < epu.rolling(window=3).mean()
            epu_panic_pulse = (epu_z > 2.5) & epu_exhaustion
            
            long_cond = long_cond | epu_panic_pulse

        # 2. FOMC 声明情绪脉冲 (NLP) - 遵守“边际变化”铁律
        if has_fomc:
            fomc = data['fomc_sentiment'].ffill()
            # 仅在边际变动的瞬间触发
            fomc_diff = fomc.diff()
            
            # 计算变动量的 Z-Score，反映预期的剧烈跳跃
            fomc_std = fomc_diff.rolling(window=252, min_periods=126).std().replace(0.0, np.nan).bfill().fillna(0.01)
            fomc_z = fomc_diff / fomc_std
            
            # 鸽派突变：大幅变鸽 (diff>0, Z>2.5)，且上次状态偏鹰派 (<0)
            fomc_dovish_pulse = (fomc_diff > 0) & (fomc_z > 2.5) & (fomc.shift(1) < 0)
            
            # 鹰派突变：大幅变鹰 (diff<0, Z<-2.5)，且上次状态偏鸽派 (>0)
            fomc_hawkish_pulse = (fomc_diff < 0) & (fomc_z < -2.5) & (fomc.shift(1) > 0)
            
            long_cond = long_cond | fomc_dovish_pulse
            short_cond = short_cond | fomc_hawkish_pulse

        # 赋予多空极值信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        # 延长信号持续周期：触发日及后2天(总共3天)，精准控制 Trigger Rate 在 5%-15% 内
        signal = signal.replace(0.0, np.nan).ffill(limit=2).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"