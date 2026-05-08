import numpy as np
import pandas as pd

class CreditVolPanicExhaustionNonlinearFactor:
    """信用与波动率恐慌极值衰竭脉冲交叉 (panic_mean_reversion/nonlinear)

    逻辑: 股市是长牛+均值回归市场，VIX或高收益债信用利差达到极端极值时往往是恐慌底部，但必须等待两者短期见顶回落（衰竭特征）才输出+1.0脉冲抄底（防接飞刀）；而在恐慌发酵初期（轻微高位且持续走阔），则是钝刀割肉的阴跌行情，输出-1.0。
    数据: vixcls (波动率), bamlh0a0hym2 (高收益债信用利差)
    输出: +1.0 表示系统性恐慌极值衰竭回落（强烈抄底），-1.0 表示恐慌情绪初期发酵恶化（看空），常态为 0.0
    触发条件: 抄底需满足VIX或利差任一Z-Score>2.0且伴随3日回落；看空需Z-Score在[0.5, 1.5]区间且近5/10日同步恶化。预期Trigger Rate约5%-15%。
    """

    def __init__(self, vix_extreme_z=2.0, spread_extreme_z=2.0, base_z=1.0, mild_z_min=0.5, mild_z_max=1.5):
        self.name = 'credit_vol_panic_exhaustion_nonlinear'
        self.vix_extreme_z = vix_extreme_z
        self.spread_extreme_z = spread_extreme_z
        self.base_z = base_z
        self.mild_z_min = mild_z_min
        self.mild_z_max = mild_z_max

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理数据缺失情况
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return pd.Series(0.0, index=data.index)

        vix = data['vixcls'].ffill()
        spread = data['bamlh0a0hym2'].ffill()

        # 计算 252日 (1年交易日) Z-Score 以衡量相对极端程度
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)

        spread_mean = spread.rolling(window=252, min_periods=60).mean()
        spread_std = spread.rolling(window=252, min_periods=60).std()
        spread_z = (spread - spread_mean) / (spread_std + 1e-8)

        # 二阶导数特征: 动量边际变化
        vix_diff_1 = vix.diff(1)
        vix_diff_3 = vix.diff(3)
        vix_diff_5 = vix.diff(5)
        vix_diff_10 = vix.diff(10)
        
        spread_diff_3 = spread.diff(3)
        spread_diff_5 = spread.diff(5)
        spread_diff_10 = spread.diff(10)

        # 默认信号输出为 0.0 (休眠状态)
        signal = pd.Series(0.0, index=data.index)

        # 1. 恐慌衰竭抄底信号 (+1.0)
        # 极度恐慌条件: 股市VIX或债市Spread至少有一个极度恐慌 (Z > 2.0)，另一个也处于高度紧张状态 (Z > 1.0)
        long_cond_extreme = ((vix_z > self.vix_extreme_z) & (spread_z > self.base_z)) | \
                            ((spread_z > self.spread_extreme_z) & (vix_z > self.base_z))
        
        # 衰竭条件 (核心防飞刀逻辑): 两者都在最近3天内停止恶化并回落，且今日VIX不创新高
        long_cond_exhaustion = (vix_diff_3 < 0) & (spread_diff_3 < 0) & (vix_diff_1 <= 0)
        
        long_signal = long_cond_extreme & long_cond_exhaustion

        # 2. 钝刀子割肉看空信号 (-1.0)
        # 温和恐慌条件: 两者均未到极致，仅处于温和发酵区 (0.5 < Z < 1.5)
        short_cond_level = (vix_z > self.mild_z_min) & (vix_z < self.mild_z_max) & \
                           (spread_z > self.mild_z_min) & (spread_z < self.mild_z_max)
                           
        # 趋势恶化条件: 5日和10日都在不断走阔(爬坡期)
        short_cond_worsening = (vix_diff_5 > 0) & (spread_diff_5 > 0) & \
                               (vix_diff_10 > 0) & (spread_diff_10 > 0) & (vix_diff_1 > 0)
                               
        short_signal = short_cond_level & short_cond_worsening

        # 赋值生成脉冲信号
        signal[long_signal] = 1.0
        signal[short_signal] = -1.0

        signal.name = self.name
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(vix_extreme_z={self.vix_extreme_z}, spread_extreme_z={self.spread_extreme_z})"