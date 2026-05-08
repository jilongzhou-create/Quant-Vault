import numpy as np
import pandas as pd

class FedCurveDislocationPulseFactor:
    """FedCurveDislocationPulseFactor (policy_pivot/nonlinear)

    逻辑: 捕捉市场对美联储政策预期(DGS2)与实际政策利率(DFEDTARU)之间的极端脱节与均值回归。当市场剧烈抢跑加息或降息时(轻微恐慌), 股指承压; 当这种极端单向预期耗尽并反转时(恐慌衰竭), 股指迎来绝佳反弹窗口。
    数据: dgs2 (2年期国债收益率), dfedtaru (联邦基金目标利率上限)
    输出: 脉冲信号, 极值衰竭看多(+1.0), 预期突变看空(-1.0)
    触发条件: Z-Score绝对值>1.5且反转看多; Z-Score正常但3日动量突变(>15bps)看空, 预期Trigger Rate ~10%
    """

    def __init__(self):
        self.name = 'fed_curve_dislocation_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        required_cols = ['dgs2', 'dfedtaru']
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        df = data[required_cols].ffill()

        # 计算市场预期与实际政策利率的脱节利差
        # 值为正: 市场预期继续加息或高位维持 (DGS2 > 联邦基金利率)
        # 值为负: 市场恐慌衰退并抢跑降息 (DGS2 < 联邦基金利率)
        spread = df['dgs2'] - df['dfedtaru']

        # 计算一年期(252个交易日)的滚动Z-Score
        rolling_mean = spread.rolling(window=252, min_periods=60).mean()
        rolling_std = spread.rolling(window=252, min_periods=60).std().replace(0, 1e-5)
        z_spread = (spread - rolling_mean) / rolling_std

        # 计算1日和3日边际动量变化
        spread_diff_1d = spread.diff()
        spread_diff_3d = spread.diff(3)

        signal = pd.Series(0.0, index=df.index)

        # 看多逻辑1: 极端加息恐慌衰竭 (买入事实/靴子落地)
        # 市场前期极度预期加息(Z>1.5), 但今日利差收窄(加息落地或通胀恐慌降温) -> 看多美股
        buy_hike_exhaustion = (z_spread > 1.5) & (spread_diff_1d < 0.0)

        # 看多逻辑2: 极端衰退/降息恐慌衰竭 (买入恐慌极值点)
        # 市场前期极度抢跑降息(Z<-1.5), 但今日利差反向扩大(降息落地或衰退恐慌消退) -> 看多美股
        buy_cut_exhaustion = (z_spread < -1.5) & (spread_diff_1d > 0.0)

        # 看空逻辑1: 突发加息冲击 (轻微恐慌 - 杀估值)
        # 从常态出发, 市场突然遭遇鹰派冲击, 3天内利差飙升超过15个基点, 且今日仍在恶化 -> 趋势看空
        sell_hike_shock = (z_spread > -1.0) & (z_spread < 1.5) & (spread_diff_3d > 0.15) & (spread_diff_1d > 0.0)

        # 看空逻辑2: 突发衰退恐慌 (轻微恐慌 - 杀盈利预期)
        # 从常态出发, 市场突然遭遇衰退恐慌(Bad news is bad news), 3天内利差暴跌超过15个基点, 且今日仍在恶化 -> 趋势看空
        sell_cut_shock = (z_spread > -1.5) & (z_spread < 1.0) & (spread_diff_3d < -0.15) & (spread_diff_1d < 0.0)

        # 分配脉冲信号
        signal[buy_hike_exhaustion | buy_cut_exhaustion] = 1.0
        signal[sell_hike_shock | sell_cut_shock] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"