import numpy as np
import pandas as pd

class UnstructuredEpuClimaxFactor:
    """经济政策不确定性(EPU)极值反转脉冲因子 (Unstructured)

    逻辑: 经济政策不确定性指数(EPU)通过NLP解析新闻文本量化宏观政策不确定性。当EPU短期内剧烈飙升(恐慌蔓延)并见顶回落时，市场避险情绪消退，资金流出安全资产，看空美债(TLT)；当EPU短期内剧烈下降(市场极度自满)并触底反弹时，避险情绪重新抬头，资金开始回流安全资产，看多美债(TLT)。
    数据: usepuindxd (美国经济政策不确定性指数)
    触发: 10日边际变化的252日Z-Score超出 ±1.5，且当前变化量出现与其动量相反的3日均值背离（即二阶导数反转的衰竭信号）。
    输出: 脉冲信号，满足极值且动能衰竭时输出 +1.0 (避险看多) 或 -1.0 (风险偏好回升看空)，其余时间为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_epu_climax_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 提取数据并前向填充，防范低频缺失
        epu = data['usepuindxd'].ffill()
        
        # 1. 结构化降噪：新闻指数天然存在日频噪音，使用5日（约1个交易周）均线进行平滑
        epu_smooth = epu.rolling(window=5, min_periods=1).mean()
        
        # 2. 边际变化(核心铁律3)：衡量近10日（两周）的不确定性变化量，捕捉突变瞬间
        epu_diff = epu_smooth.diff(10)
        
        # 3. 极值状态：计算边际变化的252日(1年期) Z-Score，判定动量是否达到极值
        epu_mean = epu_diff.rolling(window=252, min_periods=20).mean()
        epu_std = epu_diff.rolling(window=252, min_periods=20).std()
        epu_z = (epu_diff - epu_mean) / (epu_std + 1e-8)
        
        # 4. 二阶导数衰竭(核心铁律2)：当天的边际变化开始反切近3日均值，证明局部极值动能已被打破
        exhaustion_short = epu_diff < epu_diff.rolling(window=3, min_periods=1).mean()
        exhaustion_long = epu_diff > epu_diff.rolling(window=3, min_periods=1).mean()
        
        # 5. 脉冲触发(核心铁律1)
        # 看多TLT (+1.0): 市场前期极度自满(EPU大幅下降 Z < -1.5)，但下降动能衰竭并开始拐头向上，避险资金回流
        cond_buy = (epu_z < -1.5) & exhaustion_long
        
        # 看空TLT (-1.0): 市场前期极度恐慌(EPU大幅飙升 Z > 1.5)，但飙升动能衰竭并开始回落，避险溢价挤出
        cond_sell = (epu_z > 1.5) & exhaustion_short
        
        signal[cond_buy] = 1.0
        signal[cond_sell] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"