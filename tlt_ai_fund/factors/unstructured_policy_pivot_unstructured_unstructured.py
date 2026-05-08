import numpy as np
import pandas as pd

class UnstructuredPolicyPivotFactor:
    """Unstructured Policy Pivot Shock (unstructured/unstructured)

    逻辑: 捕捉美联储货币政策预期与经济政策不确定性突变引发的美债定价脉冲。由于 FOMC情绪属于低频跳跃的阶梯状非结构化数据，使用绝对水位会导致策略沦为连续做多/空的废弃因子。本因子利用 MACD 提取新闻文本情绪与政策不确定性的边际动量，严格遵循“极值 + 衰竭”二阶导原则。当鹰派恐慌或不确定性单边极致爆发后开始收缩时，意味着市场 Price-in 结束，触发美债(TLT)的抄底反转脉冲。
    数据: fomc_sentiment (FOMC鹰鸽情绪得分), usepuindxd (经济政策不确定性指数)
    触发: 情绪动量(MACD) 252日 Z-Score 达到极值 (|Z| > 2.0 ~ 2.5) 且当日动量开始反转 (MACD.diff() 改变方向)
    输出: +1.0 (鹰派极致衰竭/恐慌极致衰竭 -> 看多美债), -1.0 (鸽派极致衰竭/自满极致衰竭 -> 看空美债), 常态为 0.0
    """

    def __init__(self):
        self.name = 'unstructured_policy_pivot_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['fomc_sentiment', 'usepuindxd']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 铁律3: 边际变化 Only
        # 禁止使用绝对阶梯水位，使用 MACD (EMA5 - EMA21) 提取非结构化数据的边际动量突变
        
        # 1. FOMC 情绪边际动量 (1.0=极度鸽派, -1.0=极度鹰派)
        fomc = data['fomc_sentiment'].ffill()
        fomc_macd = fomc.ewm(span=5, adjust=False).mean() - fomc.ewm(span=21, adjust=False).mean()
        fomc_macd_mean = fomc_macd.rolling(252).mean()
        fomc_macd_std = fomc_macd.rolling(252).std().replace(0, np.nan)
        
        fomc_macd_z = ((fomc_macd - fomc_macd_mean) / fomc_macd_std).fillna(0.0)
        fomc_macd_diff = fomc_macd.diff(1)
        
        # 2. 经济政策不确定性 (EPU) 边际动量
        epu = data['usepuindxd'].ffill()
        epu_macd = epu.ewm(span=5, adjust=False).mean() - epu.ewm(span=21, adjust=False).mean()
        epu_macd_mean = epu_macd.rolling(252).mean()
        epu_macd_std = epu_macd.rolling(252).std().replace(0, np.nan)
        
        epu_macd_z = ((epu_macd - epu_macd_mean) / epu_macd_std).fillna(0.0)
        epu_macd_diff = epu_macd.diff(1)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 绝不在突发事件的第一天直接接飞刀，必须等待动能出现衰竭 (diff 反转) 才触发脉冲
        
        # 多头脉冲: 鹰派极端恐慌见顶 (动量极负但开始反弹) OR 政策极度恐慌见顶 (动量极正但开始回落)
        fomc_hawk_exhaustion = (fomc_macd_z < -2.0) & (fomc_macd_diff > 0)
        epu_panic_exhaustion = (epu_macd_z > 2.5) & (epu_macd_diff < 0)
        
        # 空头脉冲: 鸽派极端狂热见顶 (动量极正但开始收缩) OR 极度自满见底 (动量极负且开始回升)
        fomc_dove_exhaustion = (fomc_macd_z > 2.0) & (fomc_macd_diff < 0)
        epu_complacency_exhaustion = (epu_macd_z < -2.5) & (epu_macd_diff > 0)
        
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 仅在满足衰竭条件的突发短暂窗口内输出 +/-1.0，其余时间保持静默
        signal[fomc_hawk_exhaustion | epu_panic_exhaustion] = 1.0
        signal[fomc_dove_exhaustion | epu_complacency_exhaustion] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"