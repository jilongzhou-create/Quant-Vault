import numpy as np
import pandas as pd

class UnstructuredPolicyUncertaintyRateShockFactor:
    """政策不确定性与短端利率剧变非线性交叉因子 (unstructured/nonlinear)

    逻辑: 捕捉由于美联储政策剧变(Pivot)引发的美债大幅脉冲机会。当经济政策不确定性触及恐慌极值并开始回落(美联储妥协施救)，且2年期前瞻短端收益率出现极端的连续下杀并跌速放缓时(市场剧烈Price-in降息)，触发多头脉冲。符合零值休眠、二阶导数衰竭和边际变化三大铁律。
    数据: usepuindxd (经济政策不确定性指数，非结构化文本挖掘), dgs2 (2年期美债收益率)
    触发: usepuindxd Z-Score > 2.0 且回落 (恐慌衰竭) AND dgs2 5日动量 Z-Score < -2.5 且跌速趋缓 (剧烈转向边际变化)
    输出: 狙击手级脉冲信号 [+1.0/-1.0]，非触发日常态休眠为 0.0
    """

    def __init__(self, zscore_window=252, momentum_window=5, smooth_window=3):
        self.name = 'unstructured_epu_rate_shock_nonlinear'
        self.zscore_window = zscore_window
        self.momentum_window = momentum_window
        self.smooth_window = smooth_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 绝对铁律1: 初始信号为全 0.0 的 Series，实现常态休眠
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns or 'dgs2' not in data.columns:
            return signal
            
        usepu = data['usepuindxd'].ffill()
        dgs2 = data['dgs2'].ffill()

        # --- 1. 政策不确定性特征 (Unstructured EPU) ---
        usepu_mean = usepu.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).mean()
        usepu_std = usepu.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).std()
        usepu_zscore = (usepu - usepu_mean) / (usepu_std + 1e-6)
        
        # 绝对铁律2: 二阶导数衰竭 - 政策不确定性低于近期均值，代表恐慌顶峰已过，防止接飞刀
        usepu_exhaustion = usepu < usepu.rolling(window=self.smooth_window).mean()
        
        # --- 2. 短端利率特征 (dgs2) - 捕捉美联储预期剧变 ---
        # 绝对铁律3: 边际变化 - 绝对禁止直接使用绝对值，必须使用动量变化来判断资金冲击
        dgs2_diff = dgs2.diff(self.momentum_window)
        dgs2_diff_mean = dgs2_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).mean()
        dgs2_diff_std = dgs2_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).std()
        dgs2_diff_zscore = (dgs2_diff - dgs2_diff_mean) / (dgs2_diff_std + 1e-6)
        
        # 收益率暴跌动能边际减弱 (做多条件衰竭: 差值为负数且负值缩紧向上)
        dgs2_drop_exhaustion = dgs2_diff > dgs2_diff.rolling(window=self.smooth_window).mean()
        # 收益率暴涨动能边际减弱 (做空条件衰竭: 差值为正数且正值缩紧向下)
        dgs2_surge_exhaustion = dgs2_diff < dgs2_diff.rolling(window=self.smooth_window).mean()

        # --- 非线性交叉触发逻辑 ---
        
        # 做多 (Buy TLT): 宏观恐慌触顶回落 + 市场已在短端剧烈且极致地交易降息，并出现反转确认
        long_cond = (
            (usepu_zscore > 2.0) & 
            usepu_exhaustion & 
            (dgs2_diff_zscore < -2.5) & 
            dgs2_drop_exhaustion
        )

        # 做空 (Sell TLT): 宏观层面无危机状态下 + 市场突然超预期大幅上调短端利率(鹰派加息恐慌)，并出现动能收敛
        short_cond = (
            (usepu_zscore < 0.0) & 
            (dgs2_diff_zscore > 2.5) & 
            dgs2_surge_exhaustion
        )

        # 严格遵守在极值触点才输出非零信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, momentum_window={self.momentum_window})"