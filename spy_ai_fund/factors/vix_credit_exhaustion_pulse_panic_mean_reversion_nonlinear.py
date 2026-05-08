import numpy as np
import pandas as pd

class VixCreditExhaustionPulseFactor:
    """VIX与信用利差恐慌极值及衰竭交叉因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合VIX(美股恐慌)与高收益企业债利差(信用恐慌)。当两者均达到局部极度恐慌高位并同时出现边际回落时，标志着流动性冲击见顶，此时输出强看多信号(+1.0)抄底；当两者处于极度自满低位并突然同时边际上升时，标志着趋势轻微恶化，输出看空信号(-1.0)。
    数据: vixcls, bamlh0a0hym2
    输出: 脉冲信号，[-1.0, 1.0]。+1.0表示系统性恐慌衰竭产生买点，-1.0表示极度贪婪后初现恶化产生卖点。
    触发条件: 63日Z-Score大于1.5且双双见顶回落触发买入，Z-Score小于-1.0且双双触底反弹触发卖出。预期Trigger Rate 5%-15%。
    """

    def __init__(self, window=63, panic_z_vix=1.5, panic_z_hy=1.0, greed_z_vix=-1.0, greed_z_hy=-1.0):
        self.name = 'vix_credit_exhaustion_pulse'
        self.window = window
        self.panic_z_vix = panic_z_vix
        self.panic_z_hy = panic_z_hy
        self.greed_z_vix = greed_z_vix
        self.greed_z_hy = greed_z_hy

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 默认信号输出 0.0，满足零值休眠铁律
        signal = pd.Series(0.0, index=data.index)

        # 校验数据列是否完整
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            signal.name = self.name
            return signal
            
        # 提取数据并前向填充，避免因非交易日导致的数据错位
        vix = data['vixcls'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()

        # 计算 63日 (单季度) 滚动 Z-Score，反映动态的情绪基准
        vix_mean = vix.rolling(window=self.window).mean()
        vix_std = vix.rolling(window=self.window).std()
        # 避免除以零
        vix_std = vix_std.replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std

        hy_mean = hy_spread.rolling(window=self.window).mean()
        hy_std = hy_spread.rolling(window=self.window).std()
        hy_std = hy_std.replace(0, np.nan)
        hy_z = (hy_spread - hy_mean) / hy_std

        # 计算边际变化 (一阶导数动量)
        vix_diff = vix.diff()
        hy_diff = hy_spread.diff()

        # 【多头触发逻辑】二阶导数铁律: 极端恐慌 + 衰竭 = 强烈看多 (抄底)
        # 极度恐慌产生的是买点, 但必须等待恐慌见顶回落 (防止接飞刀)
        is_panic = (vix_z > self.panic_z_vix) & (hy_z > self.panic_z_hy)
        is_exhausted = (vix_diff < 0) & (hy_diff < 0)
        long_cond = is_panic & is_exhausted

        # 【空头触发逻辑】轻度恐慌爆发 = 趋势恶化 (-1.0)
        # 市场处于极度贪婪的自满状态（低VIX，低利差），但突然同时出现跃升
        is_greed = (vix_z < self.greed_z_vix) & (hy_z < self.greed_z_hy)
        is_worsening = (vix_diff > 0) & (hy_diff > 0)
        short_cond = is_greed & is_worsening

        # 赋值脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"VixCreditExhaustionPulseFactor(window={self.window})"