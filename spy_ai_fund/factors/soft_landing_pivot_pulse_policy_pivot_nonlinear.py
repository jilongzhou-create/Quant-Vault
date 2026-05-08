import numpy as np
import pandas as pd

class SoftLandingPivotPulseFactor:
    """Soft Landing Pivot Pulse Factor (policy_pivot/nonlinear)

    逻辑: 捕捉"软着陆式"宽松预期与"紧缩恐慌"的边际突变瞬间。当短端利率(2年期美债DGS2)急剧下行带动收益率曲线(T10Y2Y)极速变陡(Bull Steepening), 且高收益债信用利差(BAMLH0A0HYM2)并未走阔时, 市场定价纯粹的流动性释放而非衰退恐慌, 触发看多脉冲; 反之, 短端急升导致曲线平坦化且信用利差走阔时, 触发紧缩恐慌的看空脉冲。
    数据: [dgs2, t10y2y, bamlh0a0hym2]
    输出: 脉冲信号 [-1.0, 1.0], +1.0 看多美股(流动性宽松), -1.0 看空美股(紧缩与信用恶化)
    触发条件: 利率与利差的5日动量Z-Score突破±1.5个标准差极值, 且信用市场给出无衰退/恶化确认。预期Trigger Rate约5%-10%。
    """

    def __init__(self):
        self.name = 'soft_landing_pivot_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 检查依赖列是否存在
        required_cols = ['dgs2', 't10y2y', 'bamlh0a0hym2']
        for col in required_cols:
            if col not in data.columns:
                return signal

        # 缺失值前向填充以保证计算连续性
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()

        # 遵循【边际变化铁律】: 绝对禁止使用绝对水位, 计算5日动量变化
        dgs2_diff = dgs2.diff(5)
        t10y2y_diff = t10y2y.diff(5)
        hy_spread_diff = hy_spread.diff(5)

        # 计算252天(约1年)滚动Z-Score, 自适应不同波动率周期, 识别真正极端的瞬间
        dgs2_diff_mean = dgs2_diff.rolling(window=252, min_periods=60).mean()
        dgs2_diff_std = dgs2_diff.rolling(window=252, min_periods=60).std()
        z_dgs2_diff = (dgs2_diff - dgs2_diff_mean) / (dgs2_diff_std + 1e-8)

        t10y2y_diff_mean = t10y2y_diff.rolling(window=252, min_periods=60).mean()
        t10y2y_diff_std = t10y2y_diff.rolling(window=252, min_periods=60).std()
        z_t10y2y_diff = (t10y2y_diff - t10y2y_diff_mean) / (t10y2y_diff_std + 1e-8)

        # 看多条件: 软着陆式(Goldilocks)宽松降息抢跑
        # 1. 2年期美债收益率异常急降 (Z < -1.5)
        # 2. 收益率曲线异常急剧变陡 (Z > 1.5, 短端降幅大于长端)
        # 3. 高收益债信用利差并未走阔 (<= 0), 证明不是对经济衰退的恐慌(Hard Landing), 而是纯粹的流动性预期
        long_cond = (z_dgs2_diff < -1.5) & (z_t10y2y_diff > 1.5) & (hy_spread_diff <= 0.0)

        # 看空条件: 紧缩恐慌(Hawkish Shock)与信用恶化
        # 1. 2年期美债收益率异常急升 (Z > 1.5)
        # 2. 收益率曲线异常平坦化/倒挂加深 (Z < -1.5)
        # 3. 伴随信用利差走阔 (> 0), 证明紧缩正在引发实质性的金融压力
        short_cond = (z_dgs2_diff > 1.5) & (z_t10y2y_diff < -1.5) & (hy_spread_diff > 0.0)

        # 触发脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"