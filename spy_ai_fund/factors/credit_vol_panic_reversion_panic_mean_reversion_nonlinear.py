import numpy as np
import pandas as pd

class CreditVolPanicReversionFactor:
    """恐慌均值回归 (panic_mean_reversion/nonlinear)

    逻辑: 结合股市波动率(VIX)与信用市场高收益债利差(HY Spread)的非线性交叉。美股极度恐慌衰竭时是强烈的买点(均值回归)，而轻微恐慌的初次爆发则是下跌前兆。因子首先监控VIX与HY利差的一季度(63日)Z-Score，只有当两者同时达到历史高位且连续3日动量转负时(恐慌见顶衰竭)，才发出狙击买入信号(+1.0)；若两者偏离中枢不大但单日突发大幅跳涨，则代表趋势可能快速恶化，输出卖空信号(-1.0)。
    数据: vixcls, bamlh0a0hym2
    输出: 脉冲信号，1.0(极度恐慌衰竭/强烈看多)，-1.0(轻度恐慌爆发/趋势看空)，常态0.0。
    触发条件: 极度恐慌条件为双Z-score > 1.25且3日diff < 0；爆发恶化条件为双Z-score > 0.5且VIX日涨幅>10%、HY走阔>5bp。预期Trigger Rate在 5% - 15% 之间。
    """

    def __init__(self, z_window: int = 63, extreme_z: float = 1.25, mild_z: float = 0.5):
        self.name = 'credit_vol_panic_reversion'
        self.z_window = z_window
        self.extreme_z = extreme_z
        self.mild_z = mild_z

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 确保关键数据存在
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return pd.Series(0.0, index=data.index)

        vix = data['vixcls'].ffill()
        hy = data['bamlh0a0hym2'].ffill()

        # 计算一季度(63个交易日)滑动Z-Score识别市场情绪水位
        vix_mean = vix.rolling(self.z_window).mean()
        vix_std = vix.rolling(self.z_window).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-6)

        hy_mean = hy.rolling(self.z_window).mean()
        hy_std = hy.rolling(self.z_window).std()
        hy_z = (hy - hy_mean) / (hy_std + 1e-6)

        # 识别恐慌极值的“衰竭” (二阶导数铁律: 绝对禁止单看绝对值极值接飞刀)
        vix_diff3 = vix.diff(3)
        hy_diff3 = hy.diff(3)

        # 识别恐慌的“初次爆发” (单日急剧恶化)
        vix_ret1 = vix.pct_change(1)
        hy_diff1 = hy.diff(1)

        # 信号条件1: 极端恐慌衰竭 (抄底，强烈看多)
        # 股市与信用债市场同时在极度高位，但恐慌边际已开始缓和(回落)
        buy_cond = (
            (vix_z > self.extreme_z) &
            (hy_z > self.extreme_z) &
            (vix_diff3 < 0.0) &
            (hy_diff3 < 0.0)
        )

        # 信号条件2: 轻度恐慌爆发 (破位，强烈看空)
        # 情绪刚开始发酵(处于水面以上但未到极值点)，并伴随极高的单日斜率爆发
        sell_cond = (
            (vix_z > self.mild_z) & 
            (vix_z <= self.extreme_z) &
            (hy_z > self.mild_z) &
            (vix_ret1 > 0.10) &         # VIX单日暴涨超过10%
            (hy_diff1 > 0.05) &         # 高收益利差单日加速走阔超过5bp
            (~buy_cond)                 # 互斥
        )

        # 构建脉冲信号
        signal = pd.Series(0.0, index=data.index)
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_window={self.z_window}, extreme_z={self.extreme_z}, mild_z={self.mild_z})"