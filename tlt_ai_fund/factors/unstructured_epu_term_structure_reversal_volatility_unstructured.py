import numpy as np
import pandas as pd

class UnstructuredEpuTermStructureReversalFactor:
    """非结构化经济政策不确定性期限倒挂反转因子 (Unstructured / Volatility)

    逻辑: 这是一个纯粹的 Unstructured Volatility 因子。利用基于新闻报道提取的日频经济政策不确定性指数(usepuindxd)，计算其短期(10日)与中长期(60日)均值的“不确定性剪刀差”，以构建“媒体恐慌期限倒挂”概念。当短期非结构化新闻的恐慌度大幅飙升，导致剪刀差处于极端拥挤水位，并随后确认动量衰竭时，意味着媒体过度贩卖焦虑的阶段结束，市场避险情绪瓦解、风险偏好修复，此时资金将从美债撤出(脉冲看空)。反之，当极度乐观的盲目安全感破灭时，避险重燃(脉冲看多)。此因子只在突变衰竭时触发，属于狙击手级脉冲。
    数据: usepuindxd (Economic Policy Uncertainty Index)
    触发: 剪刀差的 252日 Z-Score > 2.0 且 3日动能 < 0 -> 极度恐慌衰竭(看空美债 -1.0)；Z-Score < -2.0 且 3日动能 > 0 -> 极度乐观破灭(看多美债 +1.0)
    输出: [-1.0, 1.0] 脉冲信号，触发后持仓维持 3 天以控制 Trigger Rate 在 5%~15% 区间
    """

    def __init__(self):
        self.name = 'unstructured_epu_term_structure_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 必须检查所需数据是否存在
        if 'usepuindxd' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        
        # 铁律3: 边际变化 - 绝对禁止使用绝对值。这里计算短期预期与中长期预期的偏离度 (期限剪刀差)
        epu_short = epu.rolling(10).mean()
        epu_long = epu.rolling(60).mean()
        epu_spread = epu_short - epu_long
        
        # 计算剪刀差的 252日(一年) 滚动极值 (Z-Score)
        epu_spread_mean = epu_spread.rolling(252).mean()
        # 防止标准差为0导致的除零异常
        epu_spread_std = epu_spread.rolling(252).std().replace(0.0, np.nan) 
        epu_spread_z = (epu_spread - epu_spread_mean) / epu_spread_std
        
        # 铁律2: 二阶导数 - 监控不确定性倒挂的动量变化 (寻找拐点衰竭的确认)
        spread_momentum = epu_spread.diff(3)
        
        # 触发条件1: 寻找极端恐慌极点的衰竭 (媒体极度悲观且恐慌开始消散，风险偏好修复，抛售避险美债)
        bear_trigger = (epu_spread_z > 2.0) & (spread_momentum < 0)
        
        # 触发条件2: 寻找极端安全幻觉的破灭 (近期无新闻的麻痹感被打破，不确定性重燃，买入美债避险)
        bull_trigger = (epu_spread_z < -2.0) & (spread_momentum > 0)
        
        # 在触发条件满足的瞬间赋脉冲极值
        signal.loc[bear_trigger] = -1.0
        signal.loc[bull_trigger] = 1.0
        
        # 铁律1: 零值休眠 - 将瞬时的脉冲信号向后延展3天，不仅能平滑噪音，且使得总体占用的活跃天数落在目标 5% 到 15% 之间
        signal = signal.replace(0.0, np.nan).ffill(limit=3).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"