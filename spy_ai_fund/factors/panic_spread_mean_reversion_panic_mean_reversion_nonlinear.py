import numpy as np
import pandas as pd

class PanicSpreadMeanReversionFactor:
    """恐慌与信用利差极值均值回归因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合VIX与美国高收益债信用利差(HY OAS)。当恐慌情绪和信用利差同时达到历史高位后开始回落时，标志着恐慌衰竭，触发强烈的抄底买入信号。如果在平静期两项指标突然同步飙升，则标志着风险偏好恶化，触发看空信号。
    数据: [vixcls, bamlh0a0hym2]
    输出: [+1.0: 恐慌极值衰竭(抄底), -1.0: 平静期恐慌急升(做空), 0.0: 常态]
    触发条件: [多头: VIX或HY Z-Score>1.8且今日下跌; 空头: 两者低位且3日内剧烈走阔。预期Trigger Rate: 5%-15%]
    """

    def __init__(self, vix_z_threshold: float = 1.8, hy_z_threshold: float = 1.8):
        self.name = 'panic_spread_mean_reversion'
        self.vix_z_threshold = vix_z_threshold
        self.hy_z_threshold = hy_z_threshold
        self.window = 126  # 使用半年窗口计算局部Z-Score

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查必需的数据列是否存在
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        vix = data['vixcls'].ffill()
        hy = data['bamlh0a0hym2'].ffill()

        # 计算 Z-Score，刻画极值状态
        vix_mean = vix.rolling(window=self.window, min_periods=30).mean()
        vix_std = vix.rolling(window=self.window, min_periods=30).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)

        hy_mean = hy.rolling(window=self.window, min_periods=30).mean()
        hy_std = hy.rolling(window=self.window, min_periods=30).std()
        hy_z = (hy - hy_mean) / hy_std.replace(0, np.nan)

        # -------------------------------------------------------------
        # 1. 抄底做多逻辑 (极值 + 衰竭)
        # -------------------------------------------------------------
        # 极度恐慌条件：任意一项达到历史相对高位
        panic_extreme = (vix_z > self.vix_z_threshold) | (hy_z > self.hy_z_threshold)
        
        # 衰竭条件：VIX今日回落，且跌破前3日均值 (确认动量扭转)，并且HY不再走阔
        vix_exhaustion = (vix.diff(1) < 0) & (vix < vix.shift(1).rolling(3).mean())
        hy_exhaustion = (hy.diff(1) <= 0)
        
        long_cond = panic_extreme & vix_exhaustion & hy_exhaustion

        # -------------------------------------------------------------
        # 2. 趋势恶化做空逻辑 (平稳 + 突发急升)
        # -------------------------------------------------------------
        # 平稳期条件：两者都在历史均值附近或以下
        calm_state = (vix_z < 0.5) & (hy_z < 0.5)
        
        # 突发急升：3天内 VIX 飙升 > 3.0 且 信用利差走阔 > 0.15
        panic_breakout = (vix.diff(3) > 3.0) & (hy.diff(3) > 0.15)
        
        short_cond = calm_state & panic_breakout

        # 生成信号
        signal = pd.Series(0.0, index=data.index)
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        # 处理可能由于滚动计算导致的 NaN 值
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(vix_z_threshold={self.vix_z_threshold}, hy_z_threshold={self.hy_z_threshold})"