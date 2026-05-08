import numpy as np
import pandas as pd

class UnstructuredFomcPivotStabilizationFactor:
    """FOMC情绪突变企稳脉冲因子 (unstructured/unstructured)

    逻辑: 捕捉美联储FOMC声明鹰鸽态度的极端突变(Marginal Pivot)。由于 FOMC 情绪得分是低频阶梯数据，
          当发生极端跳跃当天，市场往往处于剧烈的混乱定价期。为严格遵守"反接飞刀(二阶导数)"铁律，
          因子必须等待情绪跳跃"企稳(Exhaustion)"——即单日变化率归零时，确认突变冲击已落地。
          随后在极短的几天内输出顺势狙击脉冲，捕捉市场稳步消化新政策预期带来的美债趋势利润。
    数据: fomc_sentiment (NLP 鹰鸽情绪得分, 阶梯状数据)
    触发: 条件1(极值): 5日变化量(交易周)的252日 Z-Score > 2.5，且绝对变化量 > 0.1(具有经济学意义的10%结构性偏移)。
          条件2(衰竭): 1日变化量 == 0.0 (情绪指标停止跳跃，突变落地企稳)。
    输出: 极度鸽派企稳 -> 脉冲 +1.0 (看多美债)；极度鹰派企稳 -> 脉冲 -1.0 (看空美债)。非触发日严格为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_pivot_stabilization'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)

        # 数据缺失校验
        if 'fomc_sentiment' not in data.columns:
            return signal

        sentiment = data['fomc_sentiment']

        # 铁律3: 边际变化。绝对禁止使用低频情绪的绝对水位，必须使用差分捕捉预期突变。
        # delta_5d 捕捉 FOMC 会议所在周的整体情绪跳跃量
        delta_5d = sentiment.diff(5)
        
        # delta_1d 用于捕捉当天的边际动作，阶梯数据只有在会议T+1日变动，随后数日为0
        delta_1d = sentiment.diff(1)

        # 滚动 252 个交易日(自然年)计算变化量的动态均值与标准差
        roll_mean = delta_5d.rolling(window=252, min_periods=21).mean()
        
        # 标准差极小值处理，防止由于阶梯数据长期为0导致的除零错误
        roll_std = delta_5d.rolling(window=252, min_periods=21).std().replace(0.0, np.nan).fillna(1e-8)

        # 计算 Z-Score，衡量突变的极端程度
        z_score = (delta_5d - roll_mean) / roll_std

        # 条件1: 统计学极值 (Z-Score绝对值 > 2.5) + 经济学阈值 (绝对变化 > 0.1，过滤噪音微调)
        is_dovish_extreme = (z_score > 2.5) & (delta_5d > 0.1)
        is_hawkish_extreme = (z_score < -2.5) & (delta_5d < -0.1)

        # 铁律2: 二阶导数(衰竭条件)。
        # 突变发生当天 delta_1d != 0，此时处于波动率主跌/主升浪，禁止买入！
        # 等待次日 delta_1d == 0，代表"情绪跳跃已结束，指标不再恶化/变动"，此时突变企稳。
        is_exhausted = (delta_1d == 0.0)

        # 生成脉冲信号: 仅在突变且企稳的短促时间窗口内（由于 delta_5d 会保持4天的高位，信号将持续约4天）生成动作
        signal[is_dovish_extreme & is_exhausted] = 1.0   # 鸽派突变且企稳 -> 看多 TLT
        signal[is_hawkish_extreme & is_exhausted] = -1.0 # 鹰派突变且企稳 -> 看空 TLT

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"