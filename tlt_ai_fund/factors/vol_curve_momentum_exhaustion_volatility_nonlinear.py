import numpy as np
import pandas as pd

class VolCurveMomentumExhaustionFactor:
    """波动率与曲线动量非线性交叉因子 (volatility/nonlinear)

    逻辑: 将股票波动率(VIX)的极值衰竭与美债收益率曲线(10Y-2Y)的动量突变进行非线性交叉。当VIX极度恐慌且开始回落，同时收益率曲线在短期内急剧变陡(Bull Steepening)，代表市场极度出清结束且预期美联储即将降息救市，此时为强烈看多美债的绝佳狙击点；反之，极度自满(VIX极低)被打破且曲线剧烈平坦化(Bear Flattening)时，代表突发的鹰派冲击，看空美债。因子严格遵循脉冲与衰竭铁律。
    数据: vixcls (VIX波动率), t10y2y (10年期与2年期利差)
    触发: 
      看多脉冲: VIX 252日 Z-Score > 2.5 + VIX回落(二阶导) + 利差5日变陡 > 10bps (边际变化)。
      看空脉冲: VIX 252日 Z-Score < -1.5 + VIX反弹(二阶导) + 利差5日变平 < -10bps (边际变化)。
    输出: +1.0 (脉冲看多TLT), -1.0 (脉冲看空TLT), 0.0 (常态休眠)
    """

    def __init__(self):
        self.name = 'vol_curve_momentum_exhaustion_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始化全为 0.0
        signal = pd.Series(0.0, index=data.index)

        # 数据缺失校验
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            return signal

        # 前向填充处理自然缺失值
        vix = data['vixcls'].ffill()
        curve = data['t10y2y'].ffill()

        # 计算 VIX 252日滚动 Z-Score 
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 结合 1日动量(diff)与 3日均线，确保极值已实质性反转，拒绝接飞刀
        vix_rolling_3 = vix.rolling(window=3).mean()
        # 恐慌衰竭 (VIX 正在回落)
        vix_exhaustion = (vix.diff() < 0) & (vix < vix_rolling_3)
        # 自满打破 (VIX 正在反弹)
        vix_rebounding = (vix.diff() > 0) & (vix > vix_rolling_3)

        # 铁律3: 边际变化 (Marginal Change)
        # 不关注利差的绝对水位(是否倒挂)，只关注 5个交易日内的剧烈动量变化
        # t10y2y 单位为 %, 0.10 代表 5天内利差扩大 10个基点 (剧烈波动)
        curve_diff_5d = curve.diff(5)
        # Bull Steepening (牛市变陡，短端猛降，利于美债)
        curve_steepening = curve_diff_5d > 0.10
        # Bear Flattening (熊市变平，短端猛升，利空美债)
        curve_flattening = curve_diff_5d < -0.10

        # 非线性交叉组合触发
        # 多头：极度恐慌 + 恐慌开始消退 + 债市急速定价降息救市
        long_trigger = (vix_z > 2.5) & vix_exhaustion & curve_steepening
        
        # 空头：极度自满 + 波动率开始飙升 + 债市急速定价加息/紧缩
        short_trigger = (vix_z < -1.5) & vix_rebounding & curve_flattening

        # 脉冲信号赋值
        signal[long_trigger] = 1.0
        signal[short_trigger] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"