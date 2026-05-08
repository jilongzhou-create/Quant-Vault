import numpy as np
import pandas as pd

class DovishRateShockCreditPulseFactor:
    """dovish_rate_shock_credit_pulse (policy_pivot/nonlinear)

    逻辑: 捕捉美联储预期剧变引发的利率冲量。通过短端利率(DGS2)快速下行叠加曲线急剧变陡(T10Y2Y上升)的瞬间触发看多脉冲。同时通过高收益债利差(bamlh0a0hym2)没有恶化来过滤被动式衰退恐慌(防接飞刀)。相反的鹰派平坦化加信用恶化触发看空脉冲。
    数据: dgs2, t10y2y, bamlh0a0hym2
    输出: 1.0 强烈看多 (流动性变宽预期驱动的Bull Steepening)，-1.0 看空 (流动性紧缩预期驱动的平坦化/倒挂)
    触发条件: DGS2的5日动量Z-Score < -1.0 且 T10Y2Y的5日动量Z-Score > 1.0 且 HY利差无恶化。预期 Trigger Rate 5%-12%
    """

    def __init__(self, window: int = 5, z_window: int = 252, z_threshold: float = 1.0):
        self.name = 'dovish_rate_shock_credit_pulse'
        self.window = window
        self.z_window = z_window
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 必须使用的利率与信用字段
        required_cols = ['dgs2', 't10y2y', 'bamlh0a0hym2']
        
        # 处理缺失列的情况，保障鲁棒性
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        df = data[required_cols].ffill()

        # 边际变化铁律：计算 5个交易日的动量突变 (大约一周时间，捕捉脉冲)
        dgs2_diff = df['dgs2'].diff(self.window)
        t10y2y_diff = df['t10y2y'].diff(self.window)
        hy_diff = df['bamlh0a0hym2'].diff(self.window)

        # 动态波动率标准化：计算 252个交易日(约一年)滚动 Z-Score 
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(self.z_window).mean()) / dgs2_diff.rolling(self.z_window).std()
        t10y2y_z = (t10y2y_diff - t10y2y_diff.rolling(self.z_window).mean()) / t10y2y_diff.rolling(self.z_window).std()

        # 零值休眠铁律：默认为 0.0
        signal = pd.Series(0.0, index=data.index)

        # 多头脉冲条件：
        # 1. 2年期国债收益率动量剧烈下行 (市场抢跑降息, Z-Score < -1.0)
        # 2. 期限利差急剧上升变陡 (牛市变陡Bull Steepening, Z-Score > 1.0)
        # 3. 防接飞刀：高收益债利差5天内走阔不超过 5个基点(0.05)，排除宏观黑天鹅导致的衰退式暴跌
        long_cond = (dgs2_z < -self.z_threshold) & (t10y2y_z > self.z_threshold) & (hy_diff <= 0.05)
        
        # 空头脉冲条件：
        # 1. 2年期狂飙 (紧缩预期加剧, Z-Score > 1.0)
        # 2. 曲线急剧平坦化或倒挂加深 (Z-Score < -1.0)
        # 3. 高收益债利差开始走阔 (信用环境实质性恶化，流动性+信用双杀)
        short_cond = (dgs2_z > self.z_threshold) & (t10y2y_z < -self.z_threshold) & (hy_diff > 0.05)

        # 写入脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        signal.name = self.name
        
        # 抛弃 NaN 值带来的干扰
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_window={self.z_window}, z_threshold={self.z_threshold})"