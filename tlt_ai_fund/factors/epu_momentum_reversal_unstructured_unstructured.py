import numpy as np
import pandas as pd

class EpuMomentumReversalFactor:
    """经济政策不确定性短期动量反转因子 (unstructured/unstructured)

    逻辑: 政策不确定性(EPU)短期内以惊人的速度急速飙升往往对应宏观黑天鹅事件，市场陷入恐慌抛售；当不确定性见顶回落时，意味着恐慌情绪枯竭，市场开始计入联储宽松救市的预期，触发买入美债的脉冲。反之，当政策迷雾断崖式消失并企稳反弹时，风险偏好回归，做空美债。
    数据: usepuindxd (日度经济政策不确定性指数，基于新闻文本NLP)
    触发: EPU 5日变化量的 252日 Z-Score > 2.5 (极值恐慌) 且今日 EPU 下降并跌破3日均线 (二阶衰竭) -> +1.0
    输出: [-1.0, 1.0] 的狙击手级脉冲信号
    """

    def __init__(self, window=252, diff_days=5, zscore_threshold=2.5):
        self.name = 'epu_momentum_reversal'
        self.window = window
        self.diff_days = diff_days
        self.zscore_threshold = zscore_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理缺失列情况，保证常态输出为 0.0
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        epu = data['usepuindxd']
        signal = pd.Series(0.0, index=epu.index, name=self.name)

        # 铁律3: 边际变化 (只关注 EPU 的短期激增或锐减，不关注绝对水位)
        epu_diff = epu.diff(self.diff_days)
        
        # 计算 Z-Score 捕捉极端脉冲事件
        rolling_mean = epu_diff.rolling(window=self.window, min_periods=self.window // 2).mean()
        rolling_std = epu_diff.rolling(window=self.window, min_periods=self.window // 2).std()
        
        # 避免除以 0 的异常
        rolling_std = rolling_std.replace(0, np.nan)
        zscore = (epu_diff - rolling_mean) / rolling_std

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife 衰竭条件)
        # 必须等待不确定性飙升势头明确被打破才动手
        falling = (epu.diff(1) < 0) & (epu < epu.rolling(3).mean())
        rising = (epu.diff(1) > 0) & (epu > epu.rolling(3).mean())

        # 信号合成
        # 多头脉冲: 恐慌飙升达极值 (zscore > 2.5) + 动能枯竭回落 (falling)
        long_cond = (zscore > self.zscore_threshold) & falling
        
        # 空头脉冲: 恐慌断崖式消散达极值 (zscore < -2.5) + 企稳反弹风险回归 (rising)
        short_cond = (zscore < -self.zscore_threshold) & rising

        # 铁律1: 零值休眠 (仅在极端脉冲触发时赋值为 +1.0 / -1.0)
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, diff_days={self.diff_days}, z_th={self.zscore_threshold})"