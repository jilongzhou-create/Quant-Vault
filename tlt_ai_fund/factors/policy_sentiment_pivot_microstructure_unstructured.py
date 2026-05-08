import numpy as np
import pandas as pd

class NewsPanicMicrostructureFactor:
    """新闻恐慌微观流衰竭因子 (microstructure/unstructured)

    逻辑: 跟踪非结构化新闻(宏观经济政策不确定性指数 EPU)的'微观结构/信息流'拥挤极值。政策不确定性飙升常引发资产抛售与流动性溢价；当极度恐慌后不确定性开始衰竭落地，流动性修复，脉冲看多美债(TLT)。反之，极度自满(EPU骤降)后反转，代表新风险开始计价，脉冲看空美债。
    数据: usepuindxd (US Economic Policy Uncertainty Index)
    触发: EPU 边际变化或波动的 Z-Score > 2.5 且日内回落低于3日均值 -> +1.0
    输出: 狙击手级脉冲信号，精确捕捉恐慌与自满情绪的二阶导数拐点
    """

    def __init__(self):
        self.name = 'news_panic_microstructure_factor'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (Sniper Pulse)
        signal = pd.Series(0.0, index=data.index)
        
        # 数据存在性检查
        if 'usepuindxd' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        
        # 铁律3: 必须使用边际变化 (Marginal Change Only)
        epu_diff1 = epu.diff(1)
        epu_diff5 = epu.diff(5)
        epu_vol = epu_diff1.abs().rolling(5).mean()  # 新闻流量的短期波动/拥挤度
        
        # 计算多时间窗口的 Z-Score (63日和252日)，确保捕捉不同级别的脉冲以满足 5%-15% Trigger Rate
        std_5_63 = epu_diff5.rolling(63).std().replace(0, np.nan)
        std_5_252 = epu_diff5.rolling(252).std().replace(0, np.nan)
        std_vol_63 = epu_vol.rolling(63).std().replace(0, np.nan)
        
        z5_63 = (epu_diff5 - epu_diff5.rolling(63).mean()) / std_5_63
        z5_252 = (epu_diff5 - epu_diff5.rolling(252).mean()) / std_5_252
        z_vol_63 = (epu_vol - epu_vol.rolling(63).mean()) / std_vol_63
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        
        # ----------------------------------------------------
        # 看多脉冲 (+1.0): 恐慌极值 + 衰竭确认
        # ----------------------------------------------------
        # 极值条件: 过去3天内曾发生新闻恐慌量极度飙升 (任一窗口 Z-Score > 2.5)
        is_extreme_high = (z5_63 > 2.5) | (z5_252 > 2.5) | (z_vol_63 > 2.5)
        extreme_recently = is_extreme_high.rolling(window=3, min_periods=1).max().fillna(0).astype(bool)
        
        # 衰竭条件: 今天 EPU 显著回落，低于3日均值且日内边际下降
        is_exhausted = epu < epu.rolling(3).mean()
        marginal_drop = epu_diff1 < 0
        
        signal[extreme_recently & is_exhausted & marginal_drop] = 1.0
        
        # ----------------------------------------------------
        # 看空脉冲 (-1.0): 自满极值 + 反转确认
        # ----------------------------------------------------
        # 极值条件: 过去3天内曾发生新闻不确定性极度骤降 (极度自满, Z-Score < -2.5)
        is_extreme_low = (z5_63 < -2.5) | (z5_252 < -2.5)
        extreme_low_recently = is_extreme_low.rolling(window=3, min_periods=1).max().fillna(0).astype(bool)
        
        # 反转条件: 今天 EPU 显著上升，高于3日均值且日内边际上升
        is_reversing = epu > epu.rolling(3).mean()
        marginal_rise = epu_diff1 > 0
        
        signal[extreme_low_recently & is_reversing & marginal_rise] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"