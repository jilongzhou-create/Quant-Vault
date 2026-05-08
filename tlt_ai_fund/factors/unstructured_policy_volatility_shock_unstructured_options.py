import numpy as np
import pandas as pd

class UnstructuredPolicyVolatilityShockFactor:
    """Unstructured / Options

    逻辑: 捕捉经济政策不确定性(usepuindxd)的极端脉冲。政策不确定性的剧烈飙升往往迫使央行转向鸽派以对冲宏观风险，利好作为避险资产的美债。严格采用边际变化和动量衰竭条件，避免在不确定性发酵的无差别抛售期(流动性危机)过早接飞刀。
    数据: usepuindxd (每日经济政策不确定性指数)
    触发: 5日变化量的252日Z-Score > 2.5 且动量开始衰竭 (小于3日均值) 看多(+1.0)；Z-Score < -2.5 且动量跌势放缓 看空(-1.0)
    输出: 狙击手级别的脉冲型信号，常态严格为 0.0
    """

    def __init__(self, window=252, diff_days=5, exhaust_days=3, z_threshold=2.5):
        self.name = 'unstructured_policy_vol_shock'
        self.window = window            # 252个交易日，代表一整年的滚动基准
        self.diff_days = diff_days      # 5个交易日，代表单周的政策预期剧变
        self.exhaust_days = exhaust_days # 3个交易日，捕捉微观结构上的二阶导数衰竭
        self.z_threshold = z_threshold  # 2.5倍标准差，确保只有发生极端尾部事件时才触发

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (默认全0，仅在极端脉冲触发时赋值)
        signal = pd.Series(0.0, index=data.index)
        
        # 字段缺失保护
        if 'usepuindxd' not in data.columns:
            return signal

        # 获取纯域内数据
        epu_series = data['usepuindxd'].ffill()

        # 铁律3: 边际变化 (绝对禁止直接使用数据水位，必须通过变化量捕捉Shock)
        marg_change = epu_series.diff(self.diff_days)

        # 计算滚动 Z-Score 以识别尾部极端变动
        roll_mean = marg_change.rolling(window=self.window, min_periods=self.window // 2).mean()
        roll_std = marg_change.rolling(window=self.window, min_periods=self.window // 2).std()
        
        # 防止除以零
        roll_std = roll_std.replace(0.0, np.nan)
        z_score = (marg_change - roll_mean) / roll_std

        # 铁律2: 二阶导数防接飞刀 (极值必须伴随动量回落)
        # 计算短期动量均线，用于判断当前脉冲是否开始衰竭
        exhaustion_mean = marg_change.rolling(window=self.exhaust_days).mean()

        # 触发条件组装
        # 多头脉冲: 政策不确定性极端飙升 (Z > 2.5) + 飙升势头开始回落 (marg_change < 3日均值) 
        # -> 避险情绪达到高潮开始向资产定价传导，美债迎来主升浪
        long_cond = (z_score > self.z_threshold) & (marg_change < exhaustion_mean)

        # 空头脉冲: 政策不确定性超预期暴跌 (Z < -2.5) + 暴跌势头开始放缓 (marg_change > 3日均值) 
        # -> 宏观风险(如债务上限、大选)突然落地，市场无脑Risk-On，美债承压抛售
        short_cond = (z_score < -self.z_threshold) & (marg_change > exhaustion_mean)

        # 赋值非连续脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, diff_days={self.diff_days}, z_threshold={self.z_threshold})"