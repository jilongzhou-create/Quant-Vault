import numpy as np
import pandas as pd

class RepoSqueezeExhaustionFactor:
    """流动性挤兑与逆回购避险极值衰竭因子 (microstructure/nonlinear)

    逻辑: 当金融系统压力达到极值，且微观结构上大量资金涌入隔夜逆回购(ON RRP)避险导致市场失血时，美债往往处于被恐慌抛售的"接飞刀"期。只有当压力指数和逆回购规模双双见顶并开始回落(衰竭)时，才确认流动性危机解除，此时抄底美债胜率极高。
    数据: stlfsi4 (圣路易斯联储金融压力指数), rrpontsyd (隔夜逆回购量)
    触发: stlfsi4 Z-Score > 2.5 且 stlfsi4 < 3日均值 AND rrpontsyd Z-Score > 2.0 且 rrpontsyd < 3日均值
    输出: +1.0 (多重恐慌与避险囤积见顶回落，微观结构修复，美债反弹)
    """

    def __init__(self, zscore_window=252, stress_threshold=2.5, rrp_threshold=2.0, exhaust_window=3):
        self.name = 'repo_squeeze_exhaustion_micro_nonlinear'
        self.zscore_window = zscore_window
        self.stress_threshold = stress_threshold
        self.rrp_threshold = rrp_threshold
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'stlfsi4' not in data.columns or 'rrpontsyd' not in data.columns:
            return signal

        # 填充缺失值并保持时序逻辑，禁止未来函数
        stress = data['stlfsi4'].ffill()
        rrp = data['rrpontsyd'].ffill()

        # 计算Z-Score (252日代表1年期宏观周期极值)
        stress_mean = stress.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).mean()
        stress_std = stress.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).std()
        stress_zscore = (stress - stress_mean) / (stress_std + 1e-8)

        rrp_mean = rrp.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).mean()
        rrp_std = rrp.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).std()
        rrp_zscore = (rrp - rrp_mean) / (rrp_std + 1e-8)

        # 边际变化与二阶导数衰竭条件: 极值后必须开始回落
        stress_exhaustion = stress < stress.rolling(window=self.exhaust_window).mean()
        rrp_exhaustion = rrp < rrp.rolling(window=self.exhaust_window).mean()

        # 极值条件
        stress_extreme = stress_zscore > self.stress_threshold
        rrp_extreme = rrp_zscore > self.rrp_threshold

        # 交叉触发条件
        trigger = stress_extreme & stress_exhaustion & rrp_extreme & rrp_exhaustion
        
        signal.loc[trigger] = 1.0
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, stress_threshold={self.stress_threshold})"