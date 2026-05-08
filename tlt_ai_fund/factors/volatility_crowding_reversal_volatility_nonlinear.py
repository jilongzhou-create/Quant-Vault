import numpy as np
import pandas as pd

class VolatilityCrowdingReversalFactor:
    """波动率极值与拥挤反转因子 (volatility/nonlinear)

    逻辑: 监控跨资产波动率(股市VIX、黄金GVZ)与经济政策不确定性(EPU)的恐慌狂飙。绝对禁止在恐慌极值直接做多(避开流动性枯竭的主跌浪飞刀)。必须等待多领域恐慌指标处于极端高位(Z-Score>2.5)，且交叉领域的高位指标均同步开始边际回落(二阶导数衰竭)。当全面恐慌与拥挤对冲达到顶峰并开始瓦解的瞬间，长债将迎来猛烈的反转与避险配置脉冲。
    数据: vixcls, gvzcls, usepuindxd
    触发: 至少1项指标处于极度恐慌(Z>2.5)且至少2项指标跨界共振处于高位(Z>1.5)，且所有处于高位的恐慌指标当日均出现同步衰竭(diff < 0 AND < 3日均值)
    输出: +1.0 狙击看多美债脉冲，常态信号严格为 0.0
    """

    def __init__(self, z_window=126, smooth_window=3):
        self.name = 'volatility_crowding_reversal'
        self.z_window = z_window
        self.smooth_window = smooth_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (Sniper Pulse)
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'gvzcls', 'usepuindxd']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 数据前向填充，防止发布频率与节假日对齐缺失
        df = data[required_cols].ffill()
        
        # 1. 测算恐慌水位的绝对极值 (Z-Score)
        z_scores = (df - df.rolling(self.z_window).mean()) / df.rolling(self.z_window).std()
        
        # 2. 铁律3 & 铁律2: 边际变化与二阶导数衰竭确认
        # 条件: 当日指标开始回落 (diff < 0) 并且已跌破短期支撑 (< 3日均线)
        diff_neg = df.diff() < 0
        below_ma = df < df.rolling(self.smooth_window).mean()
        exhausted = diff_neg & below_ma
        
        # 3. 跨资产高位共振极值确认
        extreme_mask = z_scores > 2.5
        high_mask = z_scores > 1.5
        
        extreme_count = extreme_mask.sum(axis=1)
        high_count = high_mask.sum(axis=1)
        
        # 必须同时满足: 某个指标已经击穿绝对极值(狂飙)，且跨资产确认另一维度的恐慌也处于高位
        cross_asset_panic = (extreme_count >= 1) & (high_count >= 2)
        
        # 4. 同步衰竭反转防飞刀机制
        # 提取出所有处于高位 (Z > 1.5) 的恐慌指标，严格要求它们必须"全部"出现二阶导数衰竭！
        # 如果任一高位指标仍在逆势继续飙升，说明恐慌未充分释放，直接过滤，拒绝接飞刀。
        unexhausted_high_count = (high_mask & ~exhausted).sum(axis=1)
        panic_exhaustion_confirmed = (unexhausted_high_count == 0)
        
        # 只有在极端恐慌状态下，且所有狂飙的指标同时退潮时，产生狙击做多反转脉冲
        long_trigger = cross_asset_panic & panic_exhaustion_confirmed
        
        # 赋值非线性触发脉冲
        signal[long_trigger] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_window={self.z_window}, smooth_window={self.smooth_window})"