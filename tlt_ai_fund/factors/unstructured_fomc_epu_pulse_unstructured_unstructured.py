import numpy as np
import pandas as pd

class UnstructuredFomcEpuPulseFactor:
    """情绪与不确定性双重突变因子 (unstructured/unstructured)

    逻辑: 捕捉美联储情绪的极端边际变化，以及美国经济政策不确定性(EPU)的恐慌极值衰竭。二者皆代表政策周期的重大脉冲拐点。高不确定性衰竭或情绪剧烈转鸽均利好避险和降息预期，脉冲看多美债。
    数据: fomc_sentiment (FOMC情绪得分), usepuindxd (美国经济政策不确定性指数)
    触发: 
      1. FOMC情绪5日边际变化量的252日Z-Score > 2.5 (预期瞬间跳跃)
      2. EPU季度Z-Score > 2.5 且跌破3日均值 (恐慌极值+衰竭回落)
    输出: 狙击手级脉冲信号，常态为0.0，极端鸽派跳跃或恐慌回落时输出+1.0，反之输出-1.0。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_epu_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        # 数据缺失检查
        if 'fomc_sentiment' not in data.columns or 'usepuindxd' not in data.columns:
            return signal

        # =================================================================
        # 模块 1: 政策预期边际突变 (基于 fomc_sentiment)
        # 边际变化铁律: 针对低频阶梯数据，严格使用.diff()捕捉突变瞬间
        # =================================================================
        fomc = data['fomc_sentiment'].ffill()
        
        # 5日变化量捕捉单周内的情绪跳跃 (消化期)
        fomc_diff = fomc.diff(5)
        
        # 计算 252 日(一年)滚动 Z-Score
        fomc_diff_mean = fomc_diff.rolling(window=252, min_periods=21).mean()
        fomc_diff_std = fomc_diff.rolling(window=252, min_periods=21).std().replace(0, 1e-5)
        fomc_z = (fomc_diff - fomc_diff_mean) / fomc_diff_std
        
        # 鸽派突变: 情绪往鸽派方向急剧跳跃 -> 利多美债
        cond_fomc_dove = fomc_z > 2.5
        # 鹰派突变: 情绪往鹰派方向急剧跳跃 -> 利空美债
        cond_fomc_hawk = fomc_z < -2.5

        # =================================================================
        # 模块 2: 政策不确定性恐慌极值与衰竭 (基于 usepuindxd)
        # 二阶导数铁律: 极值 (Z-Score > 2.5) + 回落 (低于3日均值且动量为负)
        # =================================================================
        epu = data['usepuindxd'].ffill()
        
        # 计算 63 日(一季度)滚动 Z-Score
        epu_mean = epu.rolling(window=63, min_periods=10).mean()
        epu_std = epu.rolling(window=63, min_periods=10).std().replace(0, 1e-5)
        epu_z = (epu - epu_mean) / epu_std
        
        # 衰竭条件: 跌破3日均值 且 当日环比下降
        epu_exhaustion_top = (epu < epu.rolling(window=3).mean()) & (epu.diff() < 0)
        # 反弹条件: 突破3日均值 且 当日环比上升
        epu_exhaustion_bottom = (epu > epu.rolling(window=3).mean()) & (epu.diff() > 0)
        
        # 恐慌衰竭脉冲: 经济政策不确定性极度爆表后开始回落, 往往意味着美联储将被迫"救市"放水 -> 利多美债
        cond_epu_panic_exhaust = (epu_z > 2.5) & epu_exhaustion_top
        
        # 极度自满脉冲: 不确定性极低并开始抬头, 经济可能过热, 通胀担忧回归 -> 利空美债
        cond_epu_complacency_exhaust = (epu_z < -2.5) & epu_exhaustion_bottom

        # =================================================================
        # 脉冲信号合成 (常态下由于没有任何条件满足，必定保持为 0.0)
        # =================================================================
        signal[cond_fomc_dove | cond_epu_panic_exhaust] = 1.0
        signal[cond_fomc_hawk | cond_epu_complacency_exhaust] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"UnstructuredFomcEpuPulseFactor()"