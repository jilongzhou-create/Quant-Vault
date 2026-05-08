import numpy as np
import pandas as pd

class SafehavenStressExhaustionFactor:
    """避险与系统压力衰竭因子 (panic_mean_reversion/nonlinear)

    逻辑: 当股市VIX与黄金避险波动率(GVZ)同时处于极端高位且存在系统性金融压力时，若VIX见顶回落则标志流动性抛售冲击衰竭，触发均值回归看多；常态下股金波动率突然同步飙升，预示抛售潮开启看空。
    数据: vixcls, gvzcls, stlfsi4
    输出: 极度恐慌衰竭瞬间看多(+1.0), 恐慌突发瞬间看空(-1.0), 其他时间0.0
    触发条件: 恐慌极值(VIX/GVZ Z-score > 1.2且STLFSI4 > 0)且VIX日变动为负，或者常态下VIX单日飙升>2且GVZ飙升>0.5。预期Trigger Rate 6-12%
    """

    def __init__(self):
        self.name = 'safehaven_stress_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        required_cols = ['vixcls', 'gvzcls', 'stlfsi4']
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        stlfsi = data['stlfsi4'].ffill()

        # 计算252日滚动Z-Score (无未来数据)
        vix_roll_mean = vix.rolling(252).mean()
        vix_roll_std = vix.rolling(252).std()
        gvz_roll_mean = gvz.rolling(252).mean()
        gvz_roll_std = gvz.rolling(252).std()

        vix_z = (vix - vix_roll_mean) / vix_roll_std
        gvz_z = (gvz - gvz_roll_mean) / gvz_roll_std

        # 计算每日动量变化
        vix_diff = vix.diff()
        gvz_diff = gvz.diff()

        # 构建看多条件 (极值 + 衰竭)
        # 1. 股票与黄金避险情绪均处于历史高位极值 (Z-Score > 1.2)
        # 2. 金融条件处于紧缩压力区间 (STLFSI4 > 0)
        # 3. 恐慌开始回落 (vix_diff < 0)
        long_cond = (vix_z > 1.2) & (gvz_z > 1.2) & (stlfsi > 0.0) & (vix_diff < 0.0)

        # 构建看空条件 (平稳期 + 突发恐慌)
        # 1. 之前并非极端恐慌时期 (昨日VIX Z-score < 1.0)
        # 2. 股市波动率单日急剧拉升 (VIX跳涨 > 2.0)
        # 3. 避险资产波动率同步走阔验证真实恐慌 (GVZ跳涨 > 0.5)
        short_cond = (vix_z.shift(1) < 1.0) & (vix_diff > 2.0) & (gvz_diff > 0.5)

        # 合并信号
        signal = pd.Series(0.0, index=data.index)
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        # 处理可能出现的NaN
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"