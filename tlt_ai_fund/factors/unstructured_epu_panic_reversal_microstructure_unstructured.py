import numpy as np
import pandas as pd

class UnstructuredEpuPanicReversalFactor:
    """微观结构/非结构化 (EPU Panic Exhaustion)

    逻辑: 采用基于 NLP 新闻提取的美国经济政策不确定性指数(EPU, usepuindxd)，衡量宏观政策预期的微观恐慌势能。在流动性危机等极端恐慌阶段，所有资产(含TLT)往往遭到抛售；必须等待不确定性极值见顶并产生边际衰竭时，流动性压力解除，避险资金重新回流美债，触发强烈的抄底反弹。反之，当政策长期极度自满后突然反转抬头，往往预示风险积聚或紧缩担忧，触发空头抛售。此因子完全符合 Sniper Pulse 零值休眠要求，避免在单边极端市中接飞刀。
    数据: usepuindxd (基于 NLP 的经济政策不确定性日频指数)
    触发: 极高恐慌衰竭 (Z-Score > 2.5 且 当前值 < 3日均值) -> +1.0；极度自满破位 (Z-Score < -2.5 且 当前值 > 3日均值) -> -1.0。
    输出: 狙击手级脉冲信号 [-1.0, 1.0]，常态时绝对休眠为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_epu_panic_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号全为0.0，严格执行铁律1 (Sniper Pulse 零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        # 容错处理：若缺少必需字段则直接返回全0信号
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 提取并前向填充，确保无空值引发运算断层
        epu = data['usepuindxd'].ffill()
        
        # 计算微观滚动统计特征：252 个交易日窗口 (防前瞻偏差)
        rolling_mean = epu.rolling(window=252, min_periods=60).mean()
        rolling_std = epu.rolling(window=252, min_periods=60).std()
        
        # 评估当前值所处的统计极值位置，添加微小常数防止除零异常
        zscore = (epu - rolling_mean) / (rolling_std + 1e-8)
        
        # 计算短期边际状态：过去 3 日均值 (铁律3: 捕捉变化动量)
        ma3 = epu.rolling(window=3, min_periods=1).mean()
        
        # 触发条件 1: 恐慌见顶衰竭抄底脉冲 (铁律2: 二阶导数防接飞刀)
        # 条件：不确定性处于统计学极端高位，且当日值已回落跌破短期均值
        buy_pulse = (zscore > 2.5) & (epu < ma3)
        
        # 触发条件 2: 极度自满反转破位脉冲
        # 条件：不确定性处于统计学极端低位，且当日值突然抬头向上突破短期均值
        sell_pulse = (zscore < -2.5) & (epu > ma3)
        
        # 仅在触发瞬间输出极值脉冲信号
        signal.loc[buy_pulse] = 1.0
        signal.loc[sell_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"