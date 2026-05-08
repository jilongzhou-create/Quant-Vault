import numpy as np
import pandas as pd

class UnstructuredPolicyExpectationFactor:
    """政策预期与不确定性突变因子 (Unstructured)

    逻辑: 结合基于NLP解析的FOMC鹰鸽情绪边际变化与新闻提取的经济政策不确定性(EPU)指数。美债对政策预期的微小边际突变极其敏感。利用FOMC情绪得分的阶梯突变差值来捕捉央行货币政策的超预期转向；利用EPU飙升至极端且开始回落，来捕捉市场恐慌避险冲击见顶、长线资金实质性沉淀至长端美债的狙击瞬间。
    数据: fomc_sentiment (FOMC鹰鸽得分), usepuindxd (经济政策不确定性指数)
    触发: 
      1. FOMC情绪5日变化量的 Z-Score > 2.5 -> 鸽派突变脉冲 (+1.0)
      2. FOMC情绪5日变化量的 Z-Score < -2.5 -> 鹰派突变脉冲 (-1.0)
      3. EPU 水位 Z-Score > 2.5 且开始回落 (3日变化 < 0) -> 恐慌见顶衰竭看多脉冲 (+1.0)
      4. EPU 水位 Z-Score < -2.5 且开始回升 (3日变化 > 0) -> 极度安逸破灭看空脉冲 (-1.0)
    输出: [-1.0, 1.0] 的狙击手级稀疏脉冲信号
    """

    def __init__(self, window_size: int = 252):
        self.name = 'unstructured_policy_expectation_shock'
        self.window = window_size

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠。初始设定为全 0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖列是否存在，缺少则返回全 0
        required_cols = ['fomc_sentiment', 'usepuindxd']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return signal
            
        # 前向填充缺失值，防止 NaN 干扰
        df = data[required_cols].ffill()
        
        # ====================================================================
        # 1. FOMC 情绪得分突变 (NLP Policy Pivot Shock)
        # 铁律3: 边际变化。绝对禁止直接使用低频阶梯数据 fomc_sentiment 的绝对值
        # ====================================================================
        fomc_diff = df['fomc_sentiment'].diff(5)
        # 为防止长期无会议导致 std 趋近于 0 从而无限大 Z-score，给定合理下限
        fomc_std = fomc_diff.rolling(self.window).std().clip(lower=0.05)
        fomc_z = (fomc_diff - fomc_diff.rolling(self.window).mean()) / fomc_std
        
        fomc_bull = (fomc_z > 2.5) & (fomc_diff > 0)
        fomc_bear = (fomc_z < -2.5) & (fomc_diff < 0)
        
        # ====================================================================
        # 2. 经济政策不确定性冲击与衰竭 (News Sentiment Shock)
        # 铁律2: 二阶导数。绝对禁止在恐慌极高点直接买入，必须等情绪极值且出现衰竭
        # ====================================================================
        epu = df['usepuindxd']
        epu_std = epu.rolling(self.window).std().clip(lower=1.0)
        epu_z = (epu - epu.rolling(self.window).mean()) / epu_std
        
        # 衡量趋势衰竭的二阶变化
        epu_diff = epu.diff(3)
        
        # EPU 极高且动能开始向下 -> 避险恐慌顶点已过，安全资产(美债)开始实质性吸筹
        epu_bull = (epu_z > 2.5) & (epu_diff < 0)
        
        # EPU 极度低迷且开始抬头 -> 市场太平幻象破灭，重估通胀/紧缩预期
        epu_bear = (epu_z < -2.5) & (epu_diff > 0)
        
        # ====================================================================
        # 3. 脉冲信号合成
        # ====================================================================
        bull_signal = fomc_bull | epu_bull
        bear_signal = fomc_bear | epu_bear
        
        signal.loc[bull_signal] = 1.0
        signal.loc[bear_signal] = -1.0
        
        # 极端罕见的情况下多空逻辑同时触发，信号互抵
        conflict = bull_signal & bear_signal
        signal.loc[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window})"