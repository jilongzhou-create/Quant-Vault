import numpy as np
import pandas as pd

class RealYieldInflationPulseFactor:
    """实际利率与通胀预期非线性交叉因子 (policy_pivot/nonlinear)

    逻辑: 估值的核心锚是实际利率。当10年期实际利率极速暴跌且通胀预期保持稳定时，代表美联储注入了纯粹的流动性(非通缩衰退)，股市迎来看多脉冲；当实际利率急速飙升且通胀预期同步上行时，代表发生滞胀/被动紧缩双杀，股市迎来看空脉冲。
    数据: dfii10 (10年期实际利率), t10yie (10年期通胀预期)
    输出: 1.0 看多, -1.0 看空, 正常期间 0.0
    触发条件: 实际利率5日变化动量的极值(Z-Score) + 日内基点突变(3-4bps) + 通胀预期过滤衰退恐慌, 预期Trigger Rate约为 5%-10%
    """

    def __init__(self):
        self.name = 'real_yield_inflation_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需数据字段是否存在
        if 'dfii10' not in data.columns or 't10yie' not in data.columns:
            return signal
            
        # 前向填充缺失值
        dfii10 = data['dfii10'].ffill()
        t10yie = data['t10yie'].ffill()
        
        # 1. 边际变化铁律：计算 5 日的中短期政策冲量变化
        dfii10_diff5 = dfii10.diff(5)
        t10yie_diff5 = t10yie.diff(5)
        
        # 2. 捕捉发生突变瞬间的日内基点变化 (动量确认)
        dfii10_diff1 = dfii10.diff(1)
        
        # 3. 统计学极值识别：252个交易日(约1年)窗口滚动 Z-Score 识别流动性预期的极值状态
        dfii10_diff5_mean = dfii10_diff5.rolling(window=252, min_periods=60).mean()
        dfii10_diff5_std = dfii10_diff5.rolling(window=252, min_periods=60).std()
        dfii10_diff5_z = (dfii10_diff5 - dfii10_diff5_mean) / (dfii10_diff5_std + 1e-8)
        
        # 4. 零值休眠铁律与非线性交叉：组合多维极值条件
        
        # 买入脉冲 (看多流动性注入):
        # - 实际利率处于近一年的极速下行通道 (Z-Score < -1.5)
        # - 通胀预期没有发生大幅崩盘 (5日跌幅不大于 5 bps), 排除了通缩和深度衰退恐慌
        # - 当日确认实际利率继续显著下挫 (>= 3 bps 的单日降幅)
        bull_cond = (
            (dfii10_diff5_z < -1.5) & 
            (t10yie_diff5 >= -0.05) & 
            (dfii10_diff1 <= -0.03)
        )
        
        # 卖出脉冲 (看空流动性紧缩):
        # - 实际利率处于极速上行通道 (Z-Score > 2.0, 紧缩恐慌往往更为猛烈故阈值稍高)
        # - 通胀预期同时还在走高 (>= 0 bps), 典型的恶性通胀倒逼紧缩
        # - 当日确认实际利率剧烈拉升 (>= 4 bps 的单日升幅)
        bear_cond = (
            (dfii10_diff5_z > 2.0) & 
            (t10yie_diff5 >= 0.00) & 
            (dfii10_diff1 >= 0.04)
        )
        
        # 赋值狙击手级别的脉冲信号
        signal[bull_cond] = 1.0
        signal[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"