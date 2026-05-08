import numpy as np
import pandas as pd

class RateShockCreditConfirmationFactor:
    """Rate Shock & Credit Confirmation (policy_pivot/nonlinear)

    逻辑: 捕捉美联储预期突变(短端利率剧烈波动)且得到信用市场交叉验证的脉冲时刻。当2年期国债收益率极速下行(Z-Score破极值)、收益率曲线牛陡、且高收益债信用利差同向收窄时，确认实质性鸽派流动性释放，看多；反之鹰派看空。
    数据: dgs2 (2年期美债), t10y2y (期限利差), bamlh0a0hym2 (高收益债利差)
    输出: +1.0 看多 (鸽派牛陡+信用扩张), -1.0 看空 (鹰派熊平+信用收缩)
    触发条件: DGS2的5日动量Z-Score突破±1.5，伴随T10Y2Y和信用利差同向确认。仅在条件首次满足的瞬间触发(预期 Trigger Rate 5-10%)。
    """

    def __init__(self, momentum_window: int = 5, z_window: int = 252, z_threshold: float = 1.5):
        self.name = 'rate_shock_credit_confirmation_policy_pivot_nonlinear'
        self.momentum_window = momentum_window
        self.z_window = z_window
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        required_cols = ['dgs2', 't10y2y', 'bamlh0a0hym2']
        
        # 检查必需字段是否缺失
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        # 前向填充以处理偶尔缺失的日度数据(如假期)
        df = data[required_cols].ffill(limit=5)
        
        # 计算 5 个交易日(一周)的边际动量变化
        dgs2_mom = df['dgs2'].diff(self.momentum_window)
        t10y2y_mom = df['t10y2y'].diff(self.momentum_window)
        credit_mom = df['bamlh0a0hym2'].diff(self.momentum_window)

        # 计算短端利率动量的 Z-Score (使用过去一年的滚动波动率作为基准)
        dgs2_mom_std = dgs2_mom.rolling(window=self.z_window, min_periods=63).std()
        dgs2_mom_std = dgs2_mom_std.replace(0, np.nan)  # 防止除以零
        dgs2_zscore = dgs2_mom / dgs2_mom_std

        # 条件1: 鸽派突变 (Bull Steepening + Credit Easing)
        # 短端利率极速下行 + 收益率曲线牛陡 + 信用市场风险偏好回升(利差收窄)
        curr_dovish = (dgs2_zscore < -self.z_threshold) & (t10y2y_mom > 0) & (credit_mom < 0)
        
        # 条件2: 鹰派冲击 (Bear Flattening + Credit Tightening)
        # 短端利率极速上行 + 收益率曲线熊平 + 信用市场恐慌(利差走阔)
        curr_hawkish = (dgs2_zscore > self.z_threshold) & (t10y2y_mom < 0) & (credit_mom > 0)

        # 边缘变化捕捉(狙击手模式): 只在状态发生改变的第一天触发脉冲信号
        prev_dovish = curr_dovish.shift(1).fillna(False)
        prev_hawkish = curr_hawkish.shift(1).fillna(False)

        long_pulse = curr_dovish & ~prev_dovish
        short_pulse = curr_hawkish & ~prev_hawkish

        # 生成脉冲信号
        signal = pd.Series(0.0, index=data.index)
        signal[long_pulse] = 1.0
        signal[short_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(momentum_window={self.momentum_window}, z_window={self.z_window}, z_threshold={self.z_threshold})"