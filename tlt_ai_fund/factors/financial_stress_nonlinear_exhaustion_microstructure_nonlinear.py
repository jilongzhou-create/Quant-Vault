import numpy as np
import pandas as pd

class FinancialStressNonlinearExhaustionFactor:
    """金融压力非线性交叉衰竭因子 (microstructure/nonlinear)

    逻辑: 将宏观微观流动性压力(STLFSI4/NFCI)与市场恐慌情绪(VIX)进行非线性特征交叉。根据反身性原理，当流动性枯竭与市场恐慌双双达到极值并开始边际衰竭时，标志着去杠杆引发的"错杀"结束，流动性将迎来修复，此时形成高度胜率的美债抄底脉冲；而在极端宽松且自满情绪反弹时输出看空脉冲。该信号平时保持休眠，仅在极端拐点突发。
    数据: vixcls, stlfsi4 (或 nfci)
    触发: (VIX Z-Score > 2.5 或 压力指数 Z-Score > 2.0) 且 VIX < 3日均值 且 压力指数 < 5日均值。空头反之。
    输出: +1.0 表示多重恐慌见顶衰竭（看多美债脉冲），-1.0 表示自满情绪反转恶化（看空美债脉冲），常态输出 0.0。
    """

    def __init__(self, window: int = 252, vix_z_high: float = 2.5, stress_z_high: float = 2.0):
        self.name = 'financial_stress_nonlinear_exhaustion'
        self.window = window
        self.vix_z_high = vix_z_high
        self.stress_z_high = stress_z_high

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化常态休眠信号为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 校验必要数据列是否存在
        if 'vixcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        
        # 寻找可用的金融压力指数 (优先 stlfsi4, 其次 nfci)
        stress_series = None
        if 'stlfsi4' in data.columns:
            stress_series = data['stlfsi4'].ffill()
        elif 'nfci' in data.columns:
            stress_series = data['nfci'].ffill()
        else:
            return signal
            
        # 1. VIX 极值与边际衰竭计算 (日频)
        vix_mean = vix.rolling(self.window).mean()
        vix_std = vix.rolling(self.window).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        # 铁律2: 二阶导数衰竭 (当前值跌破过去3日均值代表恐慌情绪开始退潮)
        vix_exhaustion_long = vix < vix.rolling(3).mean() 
        vix_exhaustion_short = vix > vix.rolling(3).mean()
        
        # 2. 压力指数 极值与边际衰竭计算 (低频填充至日频)
        stress_mean = stress_series.rolling(self.window).mean()
        stress_std = stress_series.rolling(self.window).std()
        stress_z = (stress_series - stress_mean) / stress_std.replace(0, np.nan)
        
        # 铁律3: 边际变化 (压力指数通常是周频阶梯数据，使用5日均值比较捕捉其边际变化瞬间)
        stress_exhaustion_long = stress_series < stress_series.rolling(5).mean()
        stress_exhaustion_short = stress_series > stress_series.rolling(5).mean()
        
        # --- 多头脉冲 (Sniper Pulse - 抄底反转) ---
        # 条件1: 指标处于极端高位
        vix_extreme_panic = vix_z > self.vix_z_high
        stress_extreme_panic = stress_z > self.stress_z_high
        
        # 非线性交叉验证: 任一维度的恐慌达到极值，且两个维度的恐慌都在同步边际消退
        long_trigger = (vix_extreme_panic | stress_extreme_panic) & vix_exhaustion_long & stress_exhaustion_long
        
        # --- 空头脉冲 (Sniper Pulse - 逃顶反转) ---
        # 条件1: 处于极端自满与宽松区间
        vix_extreme_complacency = vix_z < -1.5
        stress_extreme_loose = stress_z < -1.5
        
        # 非线性交叉验证: 极度宽松自满，且开始出现双重边际恶化(反弹)
        short_trigger = (vix_extreme_complacency | stress_extreme_loose) & vix_exhaustion_short & stress_exhaustion_short
        
        # 严格遵守铁律1: 仅在触发日赋值脉冲
        signal[long_trigger] = 1.0
        signal[short_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, vix_z_high={self.vix_z_high}, stress_z_high={self.stress_z_high})"