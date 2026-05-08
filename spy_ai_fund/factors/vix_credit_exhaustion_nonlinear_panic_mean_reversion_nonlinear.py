import numpy as np
import pandas as pd

class VixCreditExhaustionNonlinearFactor:
    """VIX与信用利差恐慌衰竭交叉因子 (panic_mean_reversion/nonlinear)

    逻辑: 捕捉美股隐波(VIX)与高收益债信用利差(HY Spread)的共振极值点。
          当任一指标处于历史一年期高位(Z>1.2)，且在当天双双见顶回落的瞬间(一阶导数由正转负)，确认为恐慌衰竭(接飞刀防护)，输出强看多(+1.0)。
          当波动率与利差在中等水位(Z: 0.5~1.5)开始共振加速上行时，确认为流动性恶化的前兆，输出看空(-1.0)。
    数据: vixcls, bamlh0a0hym2
    输出: +1.0 (恐慌衰竭抄底), -1.0 (恐慌发酵看空), 0.0 (常态休眠)
    触发条件: 严格的二阶导数拐点交叉过滤，仅在转折发生的瞬间触发1天，预期 Trigger Rate 8% - 12%
    """

    def __init__(self, z_window=252, extreme_z=1.2, mild_z_lower=0.5, mild_z_upper=1.5):
        self.name = 'vix_credit_exhaustion_nonlinear'
        self.z_window = z_window
        self.extreme_z = extreme_z
        self.mild_z_lower = mild_z_lower
        self.mild_z_upper = mild_z_upper

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 数据完整性校验
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 前向填充处理节假日数据缺失
        vix = data['vixcls'].ffill()
        spread = data['bamlh0a0hym2'].ffill()

        # 计算 252日 (1年交易日) 滚动 Z-Score，反映相对所处水位
        vix_mean = vix.rolling(window=self.z_window, min_periods=60).mean()
        vix_std = vix.rolling(window=self.z_window, min_periods=60).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-6)

        spread_mean = spread.rolling(window=self.z_window, min_periods=60).mean()
        spread_std = spread.rolling(window=self.z_window, min_periods=60).std()
        spread_z = (spread - spread_mean) / (spread_std + 1e-6)

        # 计算边际变化 (一阶导数)
        vix_diff1 = vix.diff(1)
        vix_diff3 = vix.diff(3)
        spread_diff1 = spread.diff(1)
        spread_diff3 = spread.diff(3)

        signal = pd.Series(0.0, index=data.index)

        # -------------------------------------------------------------
        # 极度恐慌衰竭 (Long: +1.0)
        # 防接飞刀的二阶导数铁律:
        # 前日仍在恐慌(导数>=0)，今日终于回落(导数<0)的"第一天"
        just_turned = (vix.shift(1).diff(1) >= 0) | (spread.shift(1).diff(1) >= 0)
        
        long_cond = (
            ((vix_z > self.extreme_z) | (spread_z > self.extreme_z)) &  # 处于历史极端恐慌区间
            (vix_diff1 < 0) &                                           # 且VIX确认回落
            (spread_diff1 <= 0) &                                       # 且信用利差停止走阔
            just_turned                                                 # 严格脉冲约束: 只在拐点当天触发
        )

        # -------------------------------------------------------------
        # 轻度恐慌发酵 (Short: -1.0)
        # 捕捉流动性恶化的起始段:
        # 前天平缓，昨天开始上升，今天继续加速上升的"启动脉冲"
        short_just_started = (vix_diff1 > 0) & (vix.shift(1).diff(1) > 0) & (vix.shift(2).diff(1) <= 0)
        
        short_cond = (
            (vix_z > self.mild_z_lower) & (vix_z < self.mild_z_upper) &         # VIX 处于中高水位
            (spread_z > self.mild_z_lower) & (spread_z < self.mild_z_upper) &   # 利差同样在中高水位
            (spread_diff1 > 0) &                                                # 利差当天走阔
            (vix_diff3 > 1.0) &                                                 # 3天累计涨幅显著(>1点)
            (spread_diff3 > 0.05) &                                             # 3天利差走阔(>5个基点)
            short_just_started                                                  # 严格脉冲约束: 仅抓趋势形成的破局点
        )

        # 信号赋值
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_window={self.z_window}, extreme_z={self.extreme_z})"