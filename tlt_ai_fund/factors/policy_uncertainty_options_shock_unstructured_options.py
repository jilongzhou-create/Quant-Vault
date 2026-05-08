import numpy as np
import pandas as pd

class PolicyUncertaintyOptionsShockFactor:
    """政策不确定性与期权波动率共振冲击因子 (unstructured/options)

    逻辑: 结合基于非结构化新闻文本的经济政策不确定性(USEPUINDXD)与期权隐含波动率(VIX)的边际动量。当双重恐慌剧烈跳升且加速度见顶回落时，标志流动性冲击与政策恐慌极值已过，美联储将被迫释放鸽派预期，触发看多美债(TLT)的买入脉冲；反之过度自满反转时触发看空脉冲。因子严格依赖边际变化的二阶导数衰竭，确保常态休眠，规避接飞刀。
    数据: usepuindxd, vixcls
    触发: 5日动量合成Z-Score > 2.5 且动量开始回落 -> +1.0; Z-Score < -2.5 且动量开始反弹 -> -1.0
    输出: 脉冲型信号, [-1.0, 1.0], 正值看多美债, 负值看空美债
    """

    def __init__(self, window=252, diff_days=5, smooth_days=3):
        self.name = 'policy_uncertainty_options_shock'
        self.window = window
        self.diff_days = diff_days
        self.smooth_days = smooth_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始信号全为0.0
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['vixcls', 'usepuindxd']
        if not all(col in data.columns for col in req_cols):
            signal.name = self.name
            return signal

        # 填充缺失值
        vix = data['vixcls'].ffill()
        epu = data['usepuindxd'].ffill()
        
        # 铁律3: 边际变化，计算5日动量，捕捉预期的突变瞬间而非绝对水位
        vix_mom = vix.diff(self.diff_days)
        epu_mom = epu.diff(self.diff_days)
        
        # 计算滚动Z-Score，量化突变冲击的极端程度
        vix_z = (vix_mom - vix_mom.rolling(self.window).mean()) / vix_mom.rolling(self.window).std()
        epu_z = (epu_mom - epu_mom.rolling(self.window).mean()) / epu_mom.rolling(self.window).std()
        
        # 合成双重压力动量指标
        stress_pulse = vix_z + epu_z
        
        # 计算3日均值用于判断动量衰竭
        stress_pulse_ma = stress_pulse.rolling(self.smooth_days).mean()
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 条件1: 压力动量处于极端高位 (Z-Score和 > 2.5)
        # 条件2: 压力动量开始见顶回落 (< 3日均值)
        long_cond = (stress_pulse > 2.5) & (stress_pulse < stress_pulse_ma)
        
        # 对称逻辑: 极度自满(波动率和不确定性罕见极速下降)后开始反转抬头
        short_cond = (stress_pulse < -2.5) & (stress_pulse > stress_pulse_ma)
        
        # 仅在触发日输出脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, diff_days={self.diff_days}, smooth_days={self.smooth_days})"