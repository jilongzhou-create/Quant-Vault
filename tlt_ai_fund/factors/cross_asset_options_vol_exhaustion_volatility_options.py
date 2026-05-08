import numpy as np
import pandas as pd

class CrossAssetOptionsVolExhaustionFactor:
    """Cross-Asset Options Volatility Reversal (volatility/options)

    逻辑: 监控美股(VIX)与黄金(GVZ)隐含波动率的跨资产恐慌。极端波动率飙升时通常伴随"Dash for Cash"流动性危机(连带抛售美债)。当跨资产期权恐慌见顶并出现二阶衰竭时, 抛压结束, 避险资金重返美债。极度自满被打破时则看空。
    数据: vixcls (标普500隐含波动率), gvzcls (黄金ETF隐含波动率)
    触发: 极值条件(任一期权波动率 252日 Z-Score > 2.0) + 衰竭条件(两者 diff() < 0 且 VIX 小于 3日均值) -> 脉冲做多。极度低波(Z-Score < -1.5) + 边际抬头 -> 脉冲做空。
    输出: +1.0 表示流动性危机恐慌衰竭(做多TLT), -1.0 表示自满情绪破裂预期收紧(做空TLT), 常态返回 0.0。
    """

    def __init__(self, lookback_window=252, extreme_high_z=2.0, extreme_low_z=-1.5):
        self.name = 'cross_asset_options_vol_exhaustion'
        self.lookback = lookback_window
        self.high_z = extreme_high_z
        self.low_z = extreme_low_z

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal

        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()

        vix_mean = vix.rolling(window=self.lookback, min_periods=self.lookback // 2).mean()
        vix_std = vix.rolling(window=self.lookback, min_periods=self.lookback // 2).std()
        vix_z = (vix - vix_mean) / vix_std

        gvz_mean = gvz.rolling(window=self.lookback, min_periods=self.lookback // 2).mean()
        gvz_std = gvz.rolling(window=self.lookback, min_periods=self.lookback // 2).std()
        gvz_z = (gvz - gvz_mean) / gvz_std

        vix_falling = (vix.diff(1) < 0) & (vix < vix.rolling(window=3).mean())
        gvz_falling = (gvz.diff(1) < 0)

        vix_rising = (vix.diff(1) > 0) & (vix > vix.rolling(window=3).mean())
        gvz_rising = (gvz.diff(1) > 0)

        long_cond = ((vix_z > self.high_z) | (gvz_z > self.high_z)) & vix_falling & gvz_falling

        short_cond = (vix_z < self.low_z) & (gvz_z < self.low_z) & vix_rising & gvz_rising

        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback={self.lookback}, high_z={self.high_z}, low_z={self.low_z})"