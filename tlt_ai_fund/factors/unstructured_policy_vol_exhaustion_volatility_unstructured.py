import numpy as np
import pandas as pd

class UnstructuredPolicyVolExhaustionFactor:
    """经济政策不确定性与波动率共振衰竭脉冲因子 (volatility/unstructured)

    逻辑: 结合基于非结构化文本挖掘的经济政策不确定性(EPU)与期权隐含波动率(VIX)。当政策不确定性与市场恐慌同时狂飙至历史极值时(常伴随流动性危机，美债遭无差别错杀)，此时绝对不能接飞刀。必须等待两者同步出现边际回落(二阶导数衰竭)，标志着非理性抛售结束，避险基本面主导资金重新涌入美债，触发高胜率做多脉冲。反之，在政策与市场极度自满(低波动率)被打破的瞬间，触发做空脉冲。
    数据: usepuindxd (基于新闻文本的经济政策不确定性), vixcls (VIX指数)
    触发: Long = EPU Z-Score > 2.5 & VIX Z-Score > 2.0 且两者 diff() < 0。Short = Z-Score < -2.0 且 diff() > 0。
    输出: 严格脉冲型信号 [-1.0, 0.0, 1.0]，常态休眠。
    """

    def __init__(self):
        self.name = 'unstructured_policy_vol_exhaustion_pulse'
        self.window = 252
        self.epu_long_z_threshold = 2.5
        self.vix_long_z_threshold = 2.0
        self.short_z_threshold = -2.0

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值信号，遵循狙击手脉冲铁律
        signal = pd.Series(0.0, index=data.index)

        # 检查必要数据列是否存在
        required_cols = ['usepuindxd', 'vixcls']
        if not all(col in data.columns for col in required_cols):
            return signal

        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()

        # 计算 252 日滚动 Z-Score (防前瞻偏差)
        epu_rolling_mean = epu.rolling(window=self.window, min_periods=120).mean()
        epu_rolling_std = epu.rolling(window=self.window, min_periods=120).std()
        epu_zscore = (epu - epu_rolling_mean) / (epu_rolling_std + 1e-8)

        vix_rolling_mean = vix.rolling(window=self.window, min_periods=120).mean()
        vix_rolling_std = vix.rolling(window=self.window, min_periods=120).std()
        vix_zscore = (vix - vix_rolling_mean) / (vix_rolling_std + 1e-8)

        # 计算边际变化 (动量变化铁律: 捕捉预期的突变和拐点)
        epu_diff = epu.diff()
        vix_diff = vix.diff()

        # ---------------------------------------------------------------------
        # 核心逻辑: 二阶导数抄底铁律 (极值 + 衰竭确认)
        # ---------------------------------------------------------------------
        
        # 做多脉冲: 政策恐慌与市场恐慌处于极端高位，且今日同步开始回落 (非理性抛售瓦解)
        long_condition = (
            (epu_zscore > self.epu_long_z_threshold) & 
            (vix_zscore > self.vix_long_z_threshold) & 
            (epu_diff < 0) & 
            (vix_diff < 0)
        )

        # 做空脉冲: 政策与市场极度自满(波动率极度枯竭)，且今日同步开始抬头 (风险重估，抛售美债)
        short_condition = (
            (epu_zscore < self.short_z_threshold) & 
            (vix_zscore < self.short_z_threshold) & 
            (epu_diff > 0) & 
            (vix_diff > 0)
        )

        # 赋值脉冲信号
        signal.loc[long_condition] = 1.0
        signal.loc[short_condition] = -1.0

        # 处理可能产生的 NaN 并确保类型和命名
        signal = signal.fillna(0.0)
        signal.name = self.name

        return signal

    def __repr__(self):
        return f"UnstructuredPolicyVolExhaustionFactor(window={self.window})"