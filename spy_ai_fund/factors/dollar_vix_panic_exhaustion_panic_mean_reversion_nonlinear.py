import numpy as np
import pandas as pd

class DollarVixPanicExhaustionFactor:
    """美元流动性与VIX恐慌衰竭交叉因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合广义美元指数(代表全球避险抢筹/流动性紧缩)与VIX(股市恐慌)。当双双处于极高位并同时出现边际回落时，标志着Risk-off的宏观恐慌情绪见顶衰竭，产生高胜率的脉冲买点。而在平静期这两者双双突升，则意味着恐慌降临，触发恶化做空信号。
    数据: [dtwexbgs, vixcls]
    输出: +1.0 (宏观恐慌极值衰竭，强看多)，-1.0 (恐慌突然爆发，看空)，0.0 (常态休眠)
    触发条件: 多头(VIX Z>1.2 & USD Z>1.0 且同时回落), 空头(平静期双双跳升), 预期Trigger Rate: 5%-15%
    """

    def __init__(self):
        self.name = 'dollar_vix_panic_exhaustion_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 默认输出休眠信号全0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查是否有所需的数据列
        if 'vixcls' not in data.columns or 'dtwexbgs' not in data.columns:
            signal.name = self.name
            return signal
            
        # 填充缺失值，避免假期等导致的数据不对齐
        vix = data['vixcls'].ffill()
        usd = data['dtwexbgs'].ffill()
        
        # 1. 状态度量：计算长周期(252日, 约1年)的 Z-Score
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, 1e-5)
        
        usd_mean = usd.rolling(window=252, min_periods=60).mean()
        usd_std = usd.rolling(window=252, min_periods=60).std()
        usd_z = (usd - usd_mean) / usd_std.replace(0, 1e-5)
        
        # 2. 动量与边际变化
        # 今日动能
        vix_diff_1 = vix.diff(1)
        usd_diff_1 = usd.diff(1)
        vix_ma3 = vix.rolling(window=3).mean()
        
        # 短期爆发动能(2日)
        vix_diff_2 = vix.diff(2)
        usd_pct_2 = usd.pct_change(2)
        
        # 3. 多头逻辑 (严格遵守 二阶导数铁律: 极值 + 衰竭)
        # 美元(流动性恐慌)和VIX(资产恐慌)都在过去一年的相对高位
        is_extreme_panic = (vix_z > 1.2) & (usd_z > 1.0)
        # 恐慌开始衰竭：两者今天同步回落，且VIX已经跌破最近3天的均值
        is_exhausted = (vix_diff_1 < 0) & (usd_diff_1 < 0) & (vix < vix_ma3)
        long_cond = is_extreme_panic & is_exhausted
        
        # 4. 空头逻辑 (防范飞刀：趋势恶化 / 平静被打破)
        # 前期处于相对平静期(Z-Score < 0.5)
        is_calm_before = (vix_z.shift(2) < 0.5) & (usd_z.shift(2) < 0.5)
        # 突然双双急升 (VIX 2天飙升超过2.5，美元2天升值超过0.5%抢筹)
        is_breaking = (vix_diff_2 > 2.5) & (usd_pct_2 > 0.005)
        short_cond = is_calm_before & is_breaking
        
        # 5. 信号合成 (狙击手脉冲)
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"