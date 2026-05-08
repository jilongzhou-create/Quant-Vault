import numpy as np
import pandas as pd

class PolicyPivotYieldCurvePulseFactor:
    """因子名称: Policy Pivot Yield Curve Pulse (policy_pivot/nonlinear)

    逻辑: 捕捉由于美联储政策预期突变导致的流动性冲量。通过监控2年期国债收益率的短期剧变与收益率曲线(10Y-2Y)急剧变陡/变平的非线性交叉，识别“抢跑降息”(看多)与“加息恐慌”(看空)。同时使用高收益债利差变化进行防飞刀过滤，剔除由系统性崩溃引发的被动避险降息，仅保留纯流动性宽松带来的结构性买点。
    数据: dgs2, t10y2y, bamlh0a0hym2
    输出: +1.0 看多美股 (鸽派突变且信用平稳), -1.0 看空美股 (鹰派紧缩突变), 0.0 常态休眠
    触发条件: 2年期收益率5日变动的Z-Score达到极值，且收益率曲线5日变动Z-Score达到反向极值，排除信用利差跳升期。预期Trigger Rate 5%到15%。
    """

    def __init__(self, z_threshold: float = 1.2, diff_window: int = 5, lookback: int = 252, hy_threshold: float = 0.10):
        self.name = 'policy_pivot_yield_curve_pulse_nonlinear'
        self.z_threshold = z_threshold
        self.diff_window = diff_window
        self.lookback = lookback
        self.hy_threshold = hy_threshold  # 高收益债利差变动阈值，0.10代表10个基点

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查是否有所需字段，缺失任何一个直接返回0
        required_cols = ['dgs2', 't10y2y', 'bamlh0a0hym2']
        if not all(col in data.columns for col in required_cols):
            return pd.Series(0.0, index=data.index, name=self.name)

        # 提取数据并前向填充缺失值以防止计算差分时遇到NaN
        df = data[required_cols].ffill()

        # 计算边际短期动量变化
        dgs2_diff = df['dgs2'].diff(self.diff_window)
        t10y2y_diff = df['t10y2y'].diff(self.diff_window)
        hy_diff = df['bamlh0a0hym2'].diff(self.diff_window)

        # 使用滚动窗口计算Z-Score，识别政策预期的短期"极值跳跃"
        # 避免look-ahead bias，仅使用历史252天数据计算
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(self.lookback).mean()) / dgs2_diff.rolling(self.lookback).std()
        t10y2y_z = (t10y2y_diff - t10y2y_diff.rolling(self.lookback).mean()) / t10y2y_diff.rolling(self.lookback).std()

        # 初始化休眠信号
        signal = pd.Series(0.0, index=df.index)

        # ----------------------------------------------------
        # 多头脉冲: 鸽派突变冲量 (Bull Steepening)
        # ----------------------------------------------------
        # 1. dgs2剧烈下行: 市场押注降息抢跑 (Z < -1.2)
        # 2. t10y2y剧烈变陡: 降息定价集中在短端，发生 Bull Steepening (Z > 1.2)
        # 3. 防接飞刀: 期间高收益债利差未发生明显飙升 (< 10个基点)，证明不是因衰退恐慌导致的"被动避险降息"
        long_cond = (
            (dgs2_z < -self.z_threshold) & 
            (t10y2y_z > self.z_threshold) & 
            (hy_diff < self.hy_threshold)
        )

        # ----------------------------------------------------
        # 空头脉冲: 鹰派紧缩冲量 (Bear Flattening)
        # ----------------------------------------------------
        # 1. dgs2剧烈上行: 市场对美联储加息预期急剧升温 (Z > 1.2)
        # 2. t10y2y剧烈变平或深度倒挂: 短端利率上行速度远快于长端，发生 Bear Flattening (Z < -1.2)
        short_cond = (
            (dgs2_z > self.z_threshold) & 
            (t10y2y_z < -self.z_threshold)
        )

        # 填充脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        
        # 过滤掉因为前面NaN产生的假信号，确保未满足条件的全为0.0
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(z_threshold={self.z_threshold}, diff_window={self.diff_window}, lookback={self.lookback})"