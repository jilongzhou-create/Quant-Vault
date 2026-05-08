import numpy as np
import pandas as pd

class PolicyUncertaintyShockExhaustionFactor:
    """Policy Uncertainty Shock Exhaustion Factor (unstructured/options)

    逻辑: 当经济政策不确定性(EPU)发生极端脉冲且开始衰竭时, 市场从'情绪恐慌/狂热'转向'基本面重定价', 此时介入美债避开流动性冲击。EPU暴涨衰竭看多TLT(避险), EPU暴跌衰竭看空TLT(风险偏好修复)。
    数据: usepuindxd (经济政策不确定性指数)
    触发: EPU 3日边际变化的 252日 Z-Score > 2.0 (恐慌) 或 < -2.0 (狂热), 且绝对值相比3日均线开始反转 (衰竭)
    输出: 脉冲信号, 触发日为 +1.0 或 -1.0, 常态为 0.0
    """

    def __init__(self):
        self.name = 'policy_uncertainty_shock_exhaustion_unstructured_options'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查所需字段是否存在
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        epu = data['usepuindxd'].ffill()
        
        # 铁律3: 边际变化 (禁止直接使用绝对值, 使用3日变化量捕捉突变瞬间)
        epu_chg = epu.diff(3)
        
        # 计算动量的 252日(1年) 滚动 Z-Score
        roll_mean = epu_chg.rolling(window=252, min_periods=126).mean()
        roll_std = epu_chg.rolling(window=252, min_periods=126).std()
        
        # 避免除以 0 产生的无限大问题
        roll_std = roll_std.replace(0, np.nan)
        epu_zscore = (epu_chg - roll_mean) / roll_std
        
        # 铁律2: 二阶导数 (衰竭条件: 绝对水位反向穿透 3日均线, 确认极值动作已过峰值)
        epu_ma3 = epu.rolling(window=3).mean()
        is_falling = epu < epu_ma3
        is_rising = epu > epu_ma3
        
        # 铁律1: 零值休眠 (初始化为全 0.0)
        signal = pd.Series(0.0, index=data.index)
        
        # 条件1 & 条件2 同时满足: 极值脉冲 + 衰竭确认
        
        # 恐慌冲击衰竭: 政策不确定性飙升后开始回落 -> 避险情绪趋于理性, 资金涌入美债避险 -> +1.0
        long_cond = (epu_zscore > 2.0) & is_falling
        
        # 狂热冲击衰竭: 政策不确定性骤降后企稳反弹 -> 风险偏好大幅修复结束, 抛售美债投入风险资产 -> -1.0
        short_cond = (epu_zscore < -2.0) & is_rising
        
        # 赋值脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"