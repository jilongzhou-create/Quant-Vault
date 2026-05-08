import numpy as np
import pandas as pd

class UnstructuredEpuMicrostructureFactor:
    """News-based Economic Policy Uncertainty (EPU) Microstructure Capitulation

    逻辑: News-driven economic policy uncertainty (usepuindxd) forces microstructure capitulation in Treasury markets. 当基于新闻文本(Unstructured)提取的 EPU 指数短期飙升后开始衰竭时，由不确定性引发的恐慌抛售或无脑避险买盘(微观结构)也将枯竭，进而产生强烈的均值回归脉冲。
    数据: usepuindxd (Daily News EPU), close (TLT Price)
    触发: EPU 63日(单季度) Z-Score > 1.0 且回落到3日均值以下 (二阶衰竭), 叠加 TLT 5日收益率判定超卖/超买状态。阈值设定在 1.0 以确保能产生 5%-15% 的目标触发率。
    输出: 脉冲型 [-1.0, 1.0], 恐慌见顶引发超卖反弹 (+1.0) 或避险衰竭引发超买回落 (-1.0)。常态为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_epu_microstructure_capitulation'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 缺少核心数据则直接休眠返回 0.0
        if 'usepuindxd' not in data.columns or 'close' not in data.columns:
            return signal
            
        epu = data['usepuindxd']
        close = data['close']
        
        # ==========================================
        # 1. 边际变化与极值 (Zero-value dormancy)
        # ==========================================
        # 使用 63个交易日(约一季度)作为回溯窗口，寻找由于突发新闻导致的不确定性飙升
        epu_mean = epu.rolling(window=63, min_periods=10).mean()
        epu_std = epu.rolling(window=63, min_periods=10).std().replace(0, np.nan)
        epu_z = (epu - epu_mean) / epu_std
        
        # 为解决 0% Trigger Rate 问题，采用更合理的 1.0 标准差作为突发事件阈值
        # EPU 具有高噪音特征，Z > 1.0 能筛选出当季前 15% 的高不确定性时刻
        extreme_panic = epu_z > 1.0
        
        # ==========================================
        # 2. 二阶导数衰竭 (Anti-Catch-Falling-Knife)
        # ==========================================
        # 恐慌不确定性必须确认见顶回落（当天数值跌破过去3日均值），坚决不接飞刀
        panic_exhaustion = epu < epu.rolling(window=3, min_periods=2).mean()
        
        # ==========================================
        # 3. 微观结构状态 (Microstructure Price Action)
        # ==========================================
        # 通过 TLT 的 5日累计收益率动量，判断市场在此次恐慌中是处于超卖还是超买状态
        ret_5d = close.diff(5)
        
        # 信号合成 (两个条件均满足才触发脉冲)
        
        # 恐慌见顶 + 美债近期被抛售 (流动性冲击导致错杀) -> 抄底做多脉冲
        long_cond = extreme_panic & panic_exhaustion & (ret_5d < 0)
        
        # 恐慌见顶 + 美债近期暴涨 (无脑避险情绪引发 FOMO) -> 见顶回落做空脉冲
        short_cond = extreme_panic & panic_exhaustion & (ret_5d > 0)
        
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"