import numpy as np
import pandas as pd

class GoldilocksLiquidityPulseFactor:
    """金发姑娘流动性冲量因子 (policy_pivot/nonlinear)

    逻辑: 捕捉“好宽松”与“坏紧缩”的宏观流动性预期共振极值。当短端利率预期(DGS2)大幅下行，且企业信用利差(高收益债)同时收窄时，代表政策转松且无衰退风险的“金发姑娘”环境，市场风险偏好将飙升，发出强烈看多脉冲。反之，当利率飙升且信用利差恶化走阔时，代表杀估值与杀盈利双重打击的滞胀紧缩环境，发出看空脉冲。通过信用利差的边际收缩(衰竭)来防止在衰退性降息预期中接飞刀。
    数据: dgs2 (2年期国债收益率), bamlh0a0hym2 (高收益债信用利差)
    输出: +1.0 表示金发姑娘宽松冲量(看多), -1.0 表示滞胀紧缩冲量(看空), 常态返回 0.0
    触发条件: 2年期美债收益率与高收益债利差的5日动量，经滚动年化波动率标准化后，同时突破负向极值(-1.2和-0.8倍标准差)或正向极值(+1.2和+0.8倍标准差)，且仅在满足条件的首日触发。预期 Trigger Rate 5%-12%。
    """

    def __init__(self):
        self.name = 'goldilocks_liquidity_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 确保必需字段存在
        if 'dgs2' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        dgs2 = data['dgs2'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()

        # 计算 5 个交易日的动量边际变化
        dgs2_diff = dgs2.diff(5)
        hy_diff = hy_spread.diff(5)

        # 计算过去一年的 5 日动量波动率 (252 个交易日)，动态适应不同周期的市场常态
        dgs2_vol = dgs2_diff.rolling(window=252, min_periods=60).std()
        hy_vol = hy_diff.rolling(window=252, min_periods=60).std()

        # 避免除以 0 的情况，填补极小值以防 inf
        dgs2_vol = dgs2_vol.replace(0, np.nan).bfill().fillna(0.01)
        hy_vol = hy_vol.replace(0, np.nan).bfill().fillna(0.01)

        # 计算经过波动率标准化的边际冲击 (Shock Z-Score)
        dgs2_shock = dgs2_diff / dgs2_vol
        hy_shock = hy_diff / hy_vol

        # 初始化连续状态序列
        long_cond = (dgs2_shock < -1.2) & (hy_shock < -0.8)
        short_cond = (dgs2_shock > 1.2) & (hy_shock > 0.8)

        # 脉冲提取: 零值休眠铁律，仅在状态翻转满足条件的首日触发，避免连续产生信号
        long_pulse = long_cond & (~long_cond.shift(1).fillna(False))
        short_pulse = short_cond & (~short_cond.shift(1).fillna(False))

        # 构建输出信号
        signal = pd.Series(0.0, index=data.index)
        signal[long_pulse] = 1.0
        signal[short_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"