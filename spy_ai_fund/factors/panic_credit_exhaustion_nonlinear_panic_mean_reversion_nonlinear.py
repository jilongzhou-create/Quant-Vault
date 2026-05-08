import numpy as np
import pandas as pd

class PanicCreditExhaustionNonlinearFactor:
    """股市与信用双重恐慌衰竭因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合VIX与高收益债信用利差(HY Spread)特征，识别市场恐慌的不同阶段。当两者均处于极端高位(Z-Score>1.5)，且VIX开始回落(二阶导数为负)时，表明流动性危机与极端恐慌"极值衰竭"，触发强烈抄底做多脉冲(+1.0)。反之，如果市场原本并非极端恐慌(Z-Score<1.0)，但VIX和信用利差在短短5天内急剧走阔，表明平静期被打破且趋势刚刚恶化(钝刀割肉)，触发看空脉冲(-1.0)。
    数据: [vixcls, bamlh0a0hym2]
    输出: 脉冲信号 [-1.0, 0.0, 1.0]
    触发条件: 
      - 看多(+1): VIX & 利差 252日Z-Score > 1.5 且 今日VIX.diff() < 0 且 VIX低于过去3日均值
      - 看空(-1): VIX 252日Z-Score < 1.0 且 5日内VIX涨幅 > 20% 且 5日内利差涨幅 > 5% 且 今日VIX.diff() > 0
    预期 Trigger Rate 在 5% 到 15% 之间。
    """

    def __init__(self, window: int = 252, panic_z: float = 1.5, vix_spike_pct: float = 0.20, credit_spike_pct: float = 0.05):
        self.name = 'panic_credit_exhaustion_pulse'
        self.window = window
        self.panic_z = panic_z
        self.vix_spike_pct = vix_spike_pct
        self.credit_spike_pct = credit_spike_pct

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'bamlh0a0hym2']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        vix = data['vixcls'].ffill()
        credit = data['bamlh0a0hym2'].ffill()
        
        # 极值状态提取 (计算252日滚动Z-Score去量纲)
        vix_mean = vix.rolling(window=self.window, min_periods=63).mean()
        vix_std = vix.rolling(window=self.window, min_periods=63).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, 1e-6)
        
        credit_mean = credit.rolling(window=self.window, min_periods=63).mean()
        credit_std = credit.rolling(window=self.window, min_periods=63).std()
        credit_z = (credit - credit_mean) / credit_std.replace(0, 1e-6)
        
        # 边际变化与动量提取
        vix_diff = vix.diff(1)
        vix_3d_mean = vix.rolling(window=3).mean()
        
        vix_5d_pct = vix.pct_change(5)
        credit_5d_pct = credit.pct_change(5)
        
        # 抄底脉冲 (+1.0): 绝对高位 + 情绪开始退潮(不再接飞刀)
        long_cond = (
            (vix_z > self.panic_z) & 
            (credit_z > self.panic_z) & 
            (vix_diff < 0) & 
            (vix < vix_3d_mean)
        )
        
        # 逃顶/看空脉冲 (-1.0): 平静期被打破 + 突然的恐慌飙升(趋势恶化)
        short_cond = (
            (vix_z < 1.0) & 
            (vix_5d_pct > self.vix_spike_pct) & 
            (credit_5d_pct > self.credit_spike_pct) &
            (vix_diff > 0)
        )
        
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, panic_z={self.panic_z}, vix_spike_pct={self.vix_spike_pct})"