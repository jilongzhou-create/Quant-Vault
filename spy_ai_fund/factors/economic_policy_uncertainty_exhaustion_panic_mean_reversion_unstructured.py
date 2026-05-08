import numpy as np
import pandas as pd

class EconomicPolicyUncertaintyExhaustionFactor:
    """经济政策不确定性恐慌衰竭因子 (panic_mean_reversion/unstructured)

    逻辑: 经济政策不确定性指数(EPU)是对大量新闻文本进行NLP分析得出的非结构化恐慌指标。由于美股具有极强的均值回归特性，当EPU处于历史极端高位(Z-Score>2.0)并开始向下突破3日均线时，表明新闻层面的恐慌情绪见顶回落，触发看多抄底脉冲；当EPU从常态逐渐升温并上穿10日均线时，表明不确定性恶化发酵，对美股长牛属于“钝刀割肉”，触发看空脉冲。
    数据: [usepuindxd]
    输出: 1.0 (极端不确定性见顶回落看多), -1.0 (不确定性抬头升温看空), 0.0 (常态休眠)
    触发条件: 极值+衰竭脉冲：过去3天内Z-Score>2.0且当日下穿3日均线时输出1.0；恐慌抬头脉冲：0.5<Z-Score<1.5且当日上穿10日均线时输出-1.0。严控脉冲输出，预期Trigger Rate在5%-15%之间。
    """

    def __init__(self, zscore_window: int = 252, extreme_z: float = 2.0, mild_z_low: float = 0.5, mild_z_high: float = 1.5):
        self.name = 'epu_exhaustion_pulse'
        self.zscore_window = zscore_window
        self.extreme_z = extreme_z
        self.mild_z_low = mild_z_low
        self.mild_z_high = mild_z_high

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 数据鲁棒性保护
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        epu = data['usepuindxd'].ffill()
        
        # 1. 计算长期宏观基准 (一年期均值与标准差)
        epu_mean = epu.rolling(window=self.zscore_window, min_periods=min(self.zscore_window, 60)).mean()
        epu_std = epu.rolling(window=self.zscore_window, min_periods=min(self.zscore_window, 60)).std()
        
        # 防止除零导致无穷大
        epu_std = epu_std.replace(0, np.nan)
        zscore = (epu - epu_mean) / epu_std
        
        # 2. 计算短期微观动量 (3日代表情绪极短期突变, 10日代表短期情绪发酵)
        ma3 = epu.rolling(window=3).mean()
        ma10 = epu.rolling(window=10).mean()
        
        # 3. 极度恐慌衰竭脉冲 (+1.0)
        # 【二阶导数铁律】: 绝对禁止直接因为Z>2.0买入！必须等待向下交叉衰竭
        # 极值状态: 过去3个交易日内曾达到极度恐慌
        recent_extreme = zscore.rolling(window=3).max() > self.extreme_z
        # 衰竭拐点: EPU今日跌破3日均线 (边际恐慌退潮的瞬间)
        exhaustion_cross_down = (epu < ma3) & (epu.shift(1) >= ma3.shift(1))
        
        buy_signal = recent_extreme & exhaustion_cross_down
        
        # 4. 轻微恐慌抬头脉冲 (-1.0)
        # 状态判定: Z-Score 处于 [0.5, 1.5] 中度不确定性蔓延区域
        worsening_regime = (zscore > self.mild_z_low) & (zscore < self.mild_z_high)
        # 恶化拐点: EPU向上突破10日均线 (新的不确定性发酵起点的瞬间)
        worsening_cross_up = (epu > ma10) & (epu.shift(1) <= ma10.shift(1))
        
        sell_signal = worsening_regime & worsening_cross_up
        
        # 5. 生成休眠脉冲信号
        signal = pd.Series(0.0, index=data.index)
        signal[sell_signal] = -1.0
        signal[buy_signal] = 1.0  # 极端情况下的买点优先级高于一切
        
        # 处理异常值
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, extreme_z={self.extreme_z})"