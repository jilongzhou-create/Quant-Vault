import numpy as np
import pandas as pd

class UnstructuredPolicyFogReversalFactor:
    """Unstructured Policy Fog Reversal (非结构化政策迷雾反转脉冲)

    逻辑: 基于日常新闻提取的经济政策不确定性指数(usepuindxd)。当短期不确定性相对长期均值极度走低(市场对政策异常乐观、无视风险)，且不确定性突然开始强劲反弹时，往往预示着黑天鹅预期的发酵，触发避险资金做多美债脉冲；反之，当恐慌迷雾极度高企且见顶回落时，说明利空出尽避险消退，触发做空美债脉冲。
    数据: usepuindxd (Daily News implied Economic Policy Uncertainty)
    触发: 边际动量 Z-Score 极值 (多头 <-2.0, 空头 >2.5) + 二阶导数确认衰竭与反转 (diff 与 3日均值方向共振)
    输出: +1.0 (做多美债), -1.0 (做空美债), 常态为 0.0 (狙击手级脉冲信号)
    """

    def __init__(self):
        self.name = 'unstructured_policy_fog_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化常态零值信号
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 填充缺失值，保持日频序列的连续性
        epu = data['usepuindxd'].ffill()
        
        # 铁律3: 边际变化。绝对禁止使用EPU原始绝对值！
        # 使用 10日与60日均值差 (MACD形式) 来纯粹衡量不确定性水位的边际动量
        epu_short = epu.rolling(window=10, min_periods=1).mean()
        epu_long = epu.rolling(window=60, min_periods=1).mean()
        epu_macd = epu_short - epu_long
        
        # 计算动量的 252 日滚动 Z-Score，寻找极端情绪的错配点
        macd_mean = epu_macd.rolling(window=252, min_periods=60).mean()
        macd_std = epu_macd.rolling(window=252, min_periods=60).std()
        
        # 加微小极小值防止除以零的错误
        z_score = (epu_macd - macd_mean) / (macd_std + 1e-8)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 计算MACD的变化率，作为判定情绪极值衰竭和反转的二阶导
        macd_diff = epu_macd.diff()
        macd_diff_mean3 = macd_diff.rolling(window=3, min_periods=1).mean()
        
        # 铁律1: 零值休眠，狙击手脉冲触发
        
        # 多头条件: Z-Score 极度负值(市场极度麻痹)，且二阶导数开始强劲向上转折(动量>0且加速脱离3日均值)
        # -> 预示尾部风险即将爆发，资金准备逃入美债避险
        long_condition = (z_score < -2.0) & (macd_diff > 0) & (macd_diff > macd_diff_mean3)
        
        # 空头条件: Z-Score 极度正值(政策迷雾和恐慌极度严重)，且二阶导数开始向下回落衰竭
        # -> 预示最恐慌的时刻已经过去，避险资金将从美债撤离回流风险资产
        short_condition = (z_score > 2.5) & (macd_diff < 0) & (macd_diff < macd_diff_mean3)
        
        # 只有在极端跳跃/反转瞬间赋值，保持极高胜率脉冲
        signal.loc[long_condition] = 1.0
        signal.loc[short_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"