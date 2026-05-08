import numpy as np
import pandas as pd

class VixHighYieldSpreadReversionFactor:
    """VIX与高收益债利差极值与均值回归交叉因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合VIX与高收益债信用利差(HY OAS)。当二者同时处于历史偏高位(极端恐慌)且出现回落(近2日动量转负)时，表明恐慌情绪发生衰竭，输出看多信号(+1.0)抄底。当指标处于轻度偏高位且持续恶化(动量为正)时，恐慌正在发酵，输出看空信号(-1.0)。
    数据: vixcls (VIX), bamlh0a0hym2 (ICE BofA US High Yield Index OAS)
    输出: 强恐慌衰竭瞬间看多(+1.0)，轻度恐慌恶化看空(-1.0)，常态0.0。
    触发条件: 极值回落触发多头，中等偏高且上升触发空头。预期Trigger Rate约 8%-12%。
    """

    def __init__(self, rolling_window=252, extreme_z=1.5, mild_z=0.5, momentum_window=2):
        self.name = 'vix_hy_spread_reversion'
        self.rolling_window = rolling_window
        self.extreme_z = extreme_z
        self.mild_z = mild_z
        self.momentum_window = momentum_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        req_cols = ['vixcls', 'bamlh0a0hym2']
        missing_cols = [col for col in req_cols if col not in data.columns]
        if missing_cols:
            return pd.Series(0.0, index=data.index, name=self.name)

        df = data[req_cols].ffill()

        vix = df['vixcls']
        hy = df['bamlh0a0hym2']

        # Calculate Z-Scores (1-year rolling to capture historical extremes relative to recent regime)
        vix_mean = vix.rolling(self.rolling_window, min_periods=self.rolling_window//2).mean()
        vix_std = vix.rolling(self.rolling_window, min_periods=self.rolling_window//2).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)

        hy_mean = hy.rolling(self.rolling_window, min_periods=self.rolling_window//2).mean()
        hy_std = hy.rolling(self.rolling_window, min_periods=self.rolling_window//2).std()
        hy_z = (hy - hy_mean) / hy_std.replace(0, np.nan)

        # Calculate 2nd derivative momentum (diff) to detect exhaustion or worsening
        vix_diff = vix.diff(self.momentum_window)
        hy_diff = hy.diff(self.momentum_window)

        signal = pd.Series(0.0, index=data.index, name=self.name)

        # Condition 1: Extreme panic and exhaustion -> Buy (+1.0)
        # Avoid catching falling knives by requiring negative diff (momentum turn)
        extreme_panic = (vix_z > self.extreme_z) & (hy_z > 1.0)
        exhaustion = (vix_diff < 0) & (hy_diff < 0)
        buy_cond = extreme_panic & exhaustion

        # Condition 2: Mild panic and worsening -> Sell (-1.0)
        # Moderate elevated Z-scores and increasing panic means bleeding out
        mild_panic = (vix_z > self.mild_z) & (vix_z <= self.extreme_z) & (hy_z > self.mild_z)
        worsening = (vix_diff > 0) & (hy_diff > 0)
        sell_cond = mild_panic & worsening

        # Avoid overlapping assignments
        valid_idx = ~(buy_cond & sell_cond)
        
        signal.loc[buy_cond & valid_idx] = 1.0
        signal.loc[sell_cond & valid_idx] = -1.0

        # Replace missing with 0.0
        signal = signal.fillna(0.0)

        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(rolling_window={self.rolling_window}, extreme_z={self.extreme_z}, mild_z={self.mild_z}, momentum_window={self.momentum_window})"