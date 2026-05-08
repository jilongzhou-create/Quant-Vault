import numpy as np
import pandas as pd

class UnstructuredSentimentExhaustionFactor:
    """非结构化情绪极值衰竭因子 (panic_mean_reversion/unstructured)

    逻辑: 结合基于新闻报道的经济政策不确定性(EPU)和FOMC声明的NLP情感得分，提取纯粹的文本情绪波动。
          1. EPU衰竭(抄底): 当新闻恐慌(EPU)飙升至中期高位(Z-Score>1.5)且边际回落时，意味着极度恐慌衰竭，构成胜率极高的反弹买点。
          2. EPU恶化(看空): 当EPU稳步创新高且尚未陷入极度恐慌时，属于钝刀割肉期，顺势看空。
          3. FOMC跳变: 低频NLP声明得分出现鹰鸽超预期反转时，提供瞬间多空冲击脉冲。
    数据: usepuindxd (经济政策不确定性指数), fomc_sentiment (FOMC鹰鸽情感得分)
    输出: [-1.0, 1.0] 的多空狙击手脉冲信号
    触发条件: 满足以上任意一种瞬间突变的当日触发，预期 Trigger Rate 落在 7%~12% 之间。
    """

    def __init__(self):
        self.name = 'unstructured_sentiment_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        has_epu = 'usepuindxd' in data.columns
        has_fomc = 'fomc_sentiment' in data.columns
        
        if not has_epu and not has_fomc:
            return signal
            
        # 1. 新闻报道 EPU 极值与均值回归逻辑
        if has_epu:
            epu = data['usepuindxd'].ffill()
            
            # 使用3日平滑降低新闻指数固有的单日高噪音
            epu_3 = epu.rolling(window=3).mean()
            
            # 计算 120 个交易日的中期 Z-Score，反映近半年宏观情绪相对基准的偏离度
            epu_120_mean = epu_3.rolling(window=120).mean()
            epu_120_std = epu_3.rolling(window=120).std().replace(0, 1e-6)
            z_120 = (epu_3 - epu_120_mean) / epu_120_std
            
            # 【核心物理法则 - 防接飞刀的极度恐慌抄底】
            # Z-Score > 1.5 证明当前处于绝对恐慌高位区
            # epu_3.diff() < 0 证明恐慌预期刚刚发生边际回落，恐慌开始衰竭
            epu_exhaustion = (z_120.shift(1) > 1.5) & (epu_3.diff() < 0)
            
            # 【看空逻辑 - 钝刀割肉的轻微恐慌恶化】
            # EPU 在近10天内不断创新高，且当前情绪偏紧但未极度绝望 (0.5 < Z < 1.5)
            # 长牛中这种轻微发酵的不确定性往往对应市场的温水煮青蛙阴跌
            epu_worsening = (epu_3 >= epu_3.rolling(window=10).max()) & (z_120 > 0.5) & (z_120 <= 1.5) & (epu_3.diff() > 0)
            
            signal[epu_exhaustion] = 1.0
            signal[epu_worsening] = -1.0
            
        # 2. NLP 文本转化 FOMC 情绪极值跳变逻辑
        if has_fomc:
            fomc = data['fomc_sentiment'].ffill()
            
            # 【边际变化铁律】低频数据绝对禁止直接看绝对值！
            # 计算情感差异，捕捉开会后预期的突变点
            fomc_diff = fomc.diff()
            
            # NLP得分范围为[-1, 1]，0.3 的变化代表了强烈的边际预期转向冲击
            # 鸽派突变 (多头冲击脉冲)
            fomc_dovish_shock = fomc_diff > 0.3
            
            # 鹰派突变 (空头冲击脉冲)
            fomc_hawkish_shock = fomc_diff < -0.3
            
            # 覆盖写入：联储政策转变的权重极高，具有一票否决当天的效力
            signal[fomc_dovish_shock] = 1.0
            signal[fomc_hawkish_shock] = -1.0

        # 清理异常点
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"