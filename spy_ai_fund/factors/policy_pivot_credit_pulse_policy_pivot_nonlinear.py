import numpy as np
import pandas as pd

class PolicyPivotCreditPulseFactor:
    """政策转向与信用脉冲因子 (policy_pivot/nonlinear)

    逻辑: 捕捉短端利率急剧变化(市场重估美联储预期)与高收益债信用利差边际变化的高维非线性交叉。当信用利差处于高位且停止恶化(恐慌衰竭)，且短端利率急降导致收益率曲线陡峭化(Bull Steepening)时，触发抄底买入；当信用利差开始温和走阔，且短端利率由于通胀/鹰派惊吓而飙升(Bear Flattening)时，触发看空卖出。
    数据: dgs2(2年期美债), t10y2y(长短端利差), bamlh0a0hym2(高收益债信用利差)
    输出: +1.0 表示流动性放松预期+恐慌衰竭带来的看多；-1.0 表示鹰派惊吓+信用温和恶化带来的看空；常态输出 0.0。
    触发条件: 满足特定利率动量阈值和信用利差Z-Score及拐点条件时瞬间触发，预期Trigger Rate在5%-12%左右。
    """

    def __init__(self):
        self.name = 'policy_pivot_credit_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 所需字段检查
        required_cols = ['dgs2', 't10y2y', 'bamlh0a0hym2']
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        # 提取数据并前向填充，以防低频缺失
        df = data[required_cols].ffill()
        
        dgs2 = df['dgs2']
        t10y2y = df['t10y2y']
        hy_spread = df['bamlh0a0hym2']

        # 核心物理法则1: 计算信用利差的绝对水位 (252个交易日Z-Score，衡量恐慌极端程度)
        hy_mean = hy_spread.rolling(window=252, min_periods=126).mean()
        hy_std = hy_spread.rolling(window=252, min_periods=126).std()
        hy_zscore = (hy_spread - hy_mean) / hy_std.replace(0, np.nan)

        # 核心物理法则2: 边际动量变化 (禁止使用绝对值直接触发)
        hy_diff_3 = hy_spread.diff(3)   # 极短线信用恐慌的边际变化
        hy_diff_5 = hy_spread.diff(5)   # 周度信用恶化动量
        dgs2_diff_10 = dgs2.diff(10)    # 双周短端利率动量(市场对联储预期的剧烈重估)
        t10y2y_diff_10 = t10y2y.diff(10) # 收益率曲线动量

        signal = pd.Series(0.0, index=df.index)

        # 【多头脉冲】政策转鸽(降息预期抢跑) + 极值恐慌衰竭
        # 1. 信用利差处于历史高位 (zscore > 0.8) -> 市场处于恐慌期
        # 2. 信用恐慌开始衰竭 -> hy_diff_3 <= 0 (利差收窄)
        # 3. 市场剧烈抢跑降息 -> dgs2_diff_10 < -0.10 (2年期利率2周下跌超10个基点)
        # 4. 牛陡形成 (Bull Steepening) -> t10y2y_diff_10 > 0.0
        bull_cond = (
            (hy_zscore > 0.8) &
            (hy_diff_3 <= 0.0) &
            (dgs2_diff_10 < -0.10) &
            (t10y2y_diff_10 > 0.0)
        )

        # 【空头脉冲】鹰派惊吓 + 轻微恐慌加剧 (防飞刀原则: 绝对禁止在极值恐慌期做空)
        # 1. 信用利差处于常态或轻微压力期 -> hy_zscore 介于 -0.5 到 1.5 之间
        # 2. 信用环境边际恶化 -> hy_diff_5 > 0.10 (利差5天走阔10个基点)
        # 3. 鹰派惊吓/通胀反弹 -> dgs2_diff_10 > 0.15 (2年期利率2周飙升超15个基点)
        # 4. 熊平形成 (Bear Flattening) -> t10y2y_diff_10 < 0.0
        bear_cond = (
            (hy_zscore > -0.5) &
            (hy_zscore < 1.5) &
            (hy_diff_5 > 0.10) &
            (dgs2_diff_10 > 0.15) &
            (t10y2y_diff_10 < 0.0)
        )

        signal[bull_cond] = 1.0
        signal[bear_cond] = -1.0

        # 清理由于均值或diff产生的NaN，保障常态休眠
        signal = signal.fillna(0.0)
        signal.name = self.name

        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"