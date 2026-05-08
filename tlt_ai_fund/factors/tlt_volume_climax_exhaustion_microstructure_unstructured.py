import numpy as np
import pandas as pd

class EpuPanicExhaustionUnstructuredFactor:
    """经济政策不确定性恐慌极值与衰竭反转因子 (unstructured)

    逻辑: 经济政策不确定性(EPU)基于新闻文本NLP计算，常呈现脉冲式爆发。在极度恐慌(EPU飙升)的流动性危机期间，美债可能被无差别抛售(如2020年3月主跌浪)；当恐慌见顶并开始衰竭回落时，意味着央行救市政策开始生效，美债迎来真正的避险反弹主升浪。相反，在极度自满(EPU极低)且开始回升时，反映通胀或加息担忧抬头，做空美债。
    数据: usepuindxd (日常经济政策不确定性指数，NLP非结构化数据转化)
    触发: 126日 Z-Score > 1.5 且跌破5日均线(恐慌极值+衰竭回落)看多；Z-Score < -1.5 且突破5日均线(自满极值+恐慌抬头)看空。
    输出: 脉冲信号，+1.0看多，-1.0看空。目标触发率约 6%-10%。
    """

    def __init__(self):
        self.name = 'epu_panic_exhaustion_unstructured'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始信号全为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必要数据字段
        if 'usepuindxd' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        
        # 计算 126日 (约半年) 的滚动分布，提取边际极值
        rolling_mean = epu.rolling(window=126, min_periods=63).mean()
        rolling_std = epu.rolling(window=126, min_periods=63).std().replace(0, np.nan)
        
        epu_zscore = (epu - rolling_mean) / rolling_std
        
        # 铁律3: 边际变化，计算短期(5日)均线作为动量反转的判断面
        epu_ma5 = epu.rolling(window=5, min_periods=2).mean()
        
        # 铁律2: 二阶导数 (绝不接飞刀)
        # 多头脉冲：条件1(绝对高位 Z-score > 1.5) + 条件2(动量衰竭跌破均线，流动性挤兑结束)
        long_cond = (epu_zscore > 1.5) & (epu < epu_ma5)
        
        # 空头脉冲：条件1(绝对低位 Z-score < -1.5) + 条件2(动量反转突破均线，紧缩担忧抬头)
        short_cond = (epu_zscore < -1.5) & (epu > epu_ma5)
        
        # 狙击手级脉冲赋值
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"