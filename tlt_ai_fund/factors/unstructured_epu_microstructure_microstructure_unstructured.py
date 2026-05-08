import numpy as np
import pandas as pd

class UnstructuredEpuPanicExhaustionFactor:
    """经济政策不确定性极值衰竭因子 (microstructure/unstructured)

    逻辑: 每日经济政策不确定性(EPU)基于新闻文本刻画宏观政策恐慌度。遵循"反接飞刀"的极值衰竭抄底法则：当EPU飙升至极端高位（抛售引发流动性挤兑）并开始边际回落时，表明恐慌见顶，市场重定价降息预期，触发看多美债脉冲；当EPU处于极度自满的低位并突然向上突破时，意味着"金发姑娘"环境破裂，紧缩及通胀预期起步，触发看空美债脉冲。因子严格执行零值休眠与二阶导数动量过滤。
    数据: usepuindxd (每日经济政策不确定性指数)
    触发: 
      - 看多脉冲 (+1.0): 252日 Z-Score > 1.25 (高危极值) 且 当日值跌破3日均线 且 较昨日下跌 (恐慌衰竭)
      - 看空脉冲 (-1.0): 252日 Z-Score < -1.25 (自满极值) 且 当日值突破3日均线 且 较昨日上涨 (恐慌发酵)
    输出: [-1.0, 1.0] 的狙击手脉冲信号，常态严格为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_epu_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 常态零值休眠
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal

        # 1. 提取并前向填充非结构化文本情绪数据
        epu = data['usepuindxd'].ffill()
        
        # 2. 宏观年度视角下的极端状态衡量 (252个交易日为一年，63日为一个季度)
        epu_mean = epu.rolling(window=252, min_periods=63).mean()
        epu_std = epu.rolling(window=252, min_periods=63).std().replace(0, np.nan)
        epu_z = (epu - epu_mean) / epu_std
        
        # 3. 铁律3: 边际变化与二阶导数 (短期3日均线与单日动量)
        epu_ma3 = epu.rolling(window=3).mean()
        epu_diff = epu.diff()
        
        # 4. 铁律2: 触发看多脉冲 -> 恐慌极值 且 开始衰竭回落
        # 经济学意义：新闻极度恐慌后出现向下拐点，市场结束抛售现金化，重新计入宽货币预期
        is_panic = epu_z > 1.25
        panic_exhausting = (epu < epu_ma3) & (epu_diff < 0)
        buy_pulse = is_panic & panic_exhausting
        
        # 5. 铁律2: 触发看空脉冲 -> 自满极值 且 开始发酵恶化
        # 经济学意义：长期低不确定性导致杠杆累积，当不确定性突然跃升时，紧缩/避险抛售预期杀估值
        is_complacent = epu_z < -1.25
        complacency_reversing = (epu > epu_ma3) & (epu_diff > 0)
        sell_pulse = is_complacent & complacency_reversing
        
        # 6. 信号赋值
        signal[buy_pulse] = 1.0
        signal[sell_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"