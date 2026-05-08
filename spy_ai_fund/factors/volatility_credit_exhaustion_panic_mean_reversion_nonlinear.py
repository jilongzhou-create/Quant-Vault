import numpy as np
import pandas as pd

class VolatilityCreditExhaustionFactor:
    """恐慌极值与均值回归 (panic_mean_reversion/nonlinear)

    逻辑: 结合VIX(波动率)与高收益债信用利差(违约压力)，在系统性恐慌(双高)见顶且动量衰竭回落时抄底，在极度贪婪(双低)刚开始萌生恐慌时看空。
    数据: [vixcls, bamlh0a0hym2]
    输出: 脉冲信号，+1.0代表恐慌衰竭的狙击抄底点，-1.0代表贪婪破裂的风险厌恶点，常态为0.0。
    触发条件: VIX和HYM2的Z-score极大且3日变化率为负触发看多，Z-score极小且3日变化率为正触发看空。预期Trigger Rate约8-12%。
    """

    def __init__(self, z_window=252, diff_window=3):
        self.name = 'vol_credit_exhaustion_pulse'
        self.z_window = z_window
        self.diff_window = diff_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        req_cols = ['vixcls', 'bamlh0a0hym2']
        # 处理数据缺失，若缺少必要字段直接返回全0序列
        for col in req_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        vix = data['vixcls'].ffill()
        hym2 = data['bamlh0a0hym2'].ffill()

        # 计算252日滚动Z-Score，衡量当前恐慌水平所处的历史分位状态
        vix_z = (vix - vix.rolling(self.z_window).mean()) / vix.rolling(self.z_window).std()
        hym2_z = (hym2 - hym2.rolling(self.z_window).mean()) / hym2.rolling(self.z_window).std()

        # 计算动量变化（二阶导数：防止接飞刀，必须看到动能改变）
        vix_diff = vix.diff(self.diff_window)
        hym2_diff = hym2.diff(self.diff_window)

        # 多头信号：极端恐慌 + 恐慌衰竭 (抄底)
        # 经济学含义：波动率 > 1个标准差，利差偏紧(>0.5)，但两者在过去3天均已停止恶化并开始回落
        long_cond = (vix_z > 1.0) & (hym2_z > 0.5) & (vix_diff < 0) & (hym2_diff <= 0)

        # 空头信号：极度贪婪 + 恐慌萌生 (趋势恶化)
        # 经济学含义：波动率和利差长期处于极度舒适区(低于历史均值1个标准差)，但过去3天突然开始飙升
        short_cond = (vix_z < -1.0) & (hym2_z < -0.5) & (vix_diff > 0) & (hym2_diff > 0)

        # 构造脉冲信号
        signal = pd.Series(0.0, index=data.index)
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        return signal.rename(self.name)

    def __repr__(self):
        return f"{self.__class__.__name__}(z_window={self.z_window}, diff_window={self.diff_window})"