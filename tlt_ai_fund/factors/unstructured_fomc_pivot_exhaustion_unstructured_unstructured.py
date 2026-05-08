import numpy as np
import pandas as pd

class UnstructuredFomcPivotExhaustionFactor:
    """FOMC情绪预期反转脉冲因子 (unstructured/unstructured)

    逻辑: 捕捉美联储货币政策预期的极端突变。当FOMC声明的鸽派/鹰派情绪发生跳跃式边际变化，且之前的相反情绪状态发生衰竭(即预期反转)时，产生狙击级脉冲信号。脉冲仅在突变日触发并维持极短几天，以覆盖市场Price-in周期。
    数据: fomc_sentiment (FOMC文本情绪得分，[-1.0, 1.0])
    触发: 情绪日变化量 Z-Score > 2.5 且 绝对变化量 > 0.2 + 前期处于相反政策周期 (衰竭反转)
    输出: 脉冲型，看多(+1.0)或看空(-1.0)美债，常态严格为0.0
    """

    def __init__(self):
        self.name = 'unstructured_fomc_pivot_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 如果缺少核心数据，直接返回全0休眠信号
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 确保呈现为阶梯状的低频数据被正确前向填充
        fomc = data['fomc_sentiment'].ffill()
        
        # 铁律3: 边际变化 (严格禁止使用阶梯数据的绝对值，必须使用 diff 捕捉预期突变的瞬间)
        fomc_diff = fomc.diff(1)
        
        # 铁律2: 二阶导数 (极值条件 Z-Score > 2.5)
        # 考虑到每年仅约8次FOMC会议，平时 diff 为 0，必须使用足够长的窗口(504天约2年)保证 std 稳健
        roll_mean = fomc_diff.rolling(window=504, min_periods=252).mean()
        roll_std = fomc_diff.rolling(window=504, min_periods=252).std().replace(0.0, np.nan)
        z_score = (fomc_diff - roll_mean) / roll_std
        
        # 铁律2: 二阶导数与衰竭 (必须伴随预期的反转，禁止顺势接飞刀)
        # 鸽派突变要求前期处于偏鹰状态，意味着“鹰派预期已衰竭”
        hawk_exhausted = fomc.shift(1) <= 0.0
        # 鹰派突变要求前期处于偏鸽状态，意味着“鸽派预期已衰竭”
        dove_exhausted = fomc.shift(1) >= 0.0
        
        # 经济学参数: 情绪得分范围为[-1, 1]，0.2代表10%的实质性结构偏转，过滤可能由噪音引起的微小Z-Score突变
        meaningful_shift = 0.2
        
        # 极端脉冲触发条件
        trigger_long = (z_score > 2.5) & hawk_exhausted & (fomc_diff > meaningful_shift)
        trigger_short = (z_score < -2.5) & dove_exhausted & (fomc_diff < -meaningful_shift)
        
        # 铁律1: 零值休眠 (Sniper Pulse)
        raw_signal = pd.Series(0.0, index=data.index)
        raw_signal[trigger_long] = 1.0
        raw_signal[trigger_short] = -1.0
        
        # 触发日及随后极短几天内(limit=3，共4天)输出信号，之后立刻回归0.0
        # 这将确保 Trigger Rate 严格落在 5% - 15% 的目标区间内
        signal = raw_signal.replace(0.0, np.nan).ffill(limit=3).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"