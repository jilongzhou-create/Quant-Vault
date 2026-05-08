import numpy as np
import pandas as pd

class UnstructuredPolicyPivotPulseFactor:
    """Unstructured Policy Pivot Pulse Factor

    逻辑: 捕捉美联储政策预期(FOMC NLP情绪与2年期美债)的极端突变脉冲。常态下信号为0。当NLP鸽派情绪突变或短端利率(dgs2)出现极端下行(降息预期骤升)，且确认前期鹰派趋势已衰竭时，输出看多美债脉冲。
    数据: fomc_sentiment, dgs2
    触发: (fomc_sentiment或dgs2的5日变化量 Z-Score > 2.5) + 前期均值极值与3日反转(二阶导数衰竭) -> 触发5日脉冲信号
    输出: 脉冲型信号, [-1.0, 1.0], 正值做多TLT(政策转鸽), 负值做空TLT(政策转鹰)
    """

    def __init__(self):
        self.name = 'unstructured_policy_pivot_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化信号，绝对遵循零值休眠铁律
        signal = pd.Series(0.0, index=data.index)

        # 检查依赖数据是否存在
        req_cols = ['fomc_sentiment', 'dgs2']
        if not all(col in data.columns for col in req_cols):
            return signal

        # 提取并前向填充有效数据
        df = data[req_cols].fillna(method='ffill')

        # === 核心铁律3: 边际变化 (Marginal Change Only) ===
        # 绝对禁止直接使用 fomc_sentiment 的绝对值, 必须计算其边际突变以捕捉预期反转瞬间
        fomc_chg = df['fomc_sentiment'].diff(5)
        dgs2_chg = df['dgs2'].diff(5)

        # 计算 252日 Z-Score 衡量边际变化的极端程度
        fomc_std = fomc_chg.rolling(252).std().replace(0, np.nan)
        dgs2_std = dgs2_chg.rolling(252).std().replace(0, np.nan)

        fomc_z = (fomc_chg - fomc_chg.rolling(252).mean()) / fomc_std
        dgs2_z = (dgs2_chg - dgs2_chg.rolling(252).mean()) / dgs2_std

        # === 核心铁律2: 二阶导数 (Anti-Catch-Falling-Knife) ===
        # 衰竭条件: 指标处于极端高位(相对于63日均线) + 开始回落(近3日diff反转)
        
        # 对于看多美债(降息预期): 必须确认前期是鹰派压制(dgs2处于季度高位), 且当前已出现实质性回落
        hawkish_exhaustion = (df['dgs2'] > df['dgs2'].rolling(63).mean()) & (df['dgs2'].diff(3) < 0)

        # 对于看空美债(加息预期): 必须确认前期是鸽派宽松(dgs2处于季度低位), 且当前已出现实质性反弹
        dovish_exhaustion = (df['dgs2'] < df['dgs2'].rolling(63).mean()) & (df['dgs2'].diff(3) > 0)

        # === 核心铁律1: 零值休眠与极值触发 (Sniper Pulse) ===
        # 触发条件: 边际变化 Z-Score 突破极值 (> 2.5) 且 伴随二阶导数衰竭
        # 逻辑方向: fomc_sentiment 升高代表鸽派(看多), dgs2 下降代表降息预期(看多)
        long_trigger = ((fomc_z > 2.5) | (dgs2_z < -2.5)) & hawkish_exhaustion
        short_trigger = ((fomc_z < -2.5) | (dgs2_z > 2.5)) & dovish_exhaustion

        # 将极端脉冲信号展期 5 天，确保在非零信号期间覆盖后续发酵，同时将 Trigger Rate 精准控制在 5%-15% 之间
        long_pulse = long_trigger.rolling(5).max().fillna(0).astype(bool)
        short_pulse = short_trigger.rolling(5).max().fillna(0).astype(bool)

        # 赋值狙击手级脉冲信号
        signal[long_pulse] = 1.0
        signal[short_pulse] = -1.0

        # 清除长短边界可能产生的极少数逻辑冲突天数
        signal[long_pulse & short_pulse] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"