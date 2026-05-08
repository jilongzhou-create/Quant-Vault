import numpy as np
import pandas as pd

class FomcPivotPulseFactor:
    """FOMC 政策转向流动性冲量因子 (policy_pivot/unstructured)

    逻辑: 捕捉美联储公开市场委员会(FOMC)会议声明情绪的边际剧变。当声明由鹰派大幅转为鸽派时，政策流动性预期改善，形成多头脉冲；反之鹰派突变形成空头脉冲。由于 fomc_sentiment 为低频更新的阶梯状数据，使用 5 日差分能够将其天然转化为会议后持续约 5 个交易日的脉冲方波，极好地锁定了"预期发生剧变的极短窗口"。
    数据: [fomc_sentiment]
    输出: [-1.0, 1.0] 脉冲信号。1.0代表鸽派突变看多美股，-1.0代表鹰派突变看空美股，无会议突变期间默认休眠(0.0)。
    触发条件: 情绪得分5日向上跳跃 > 0.25 或由鹰转鸽(负转正且跳跃 > 0.15)时触发 +1.0脉冲；向下跳跃 < -0.25 或由鸽转鹰(正转负且跳跃 < -0.15)时触发 -1.0脉冲。预期 Trigger Rate 控制在 5%-10% 左右。
    """

    def __init__(self):
        self.name = 'fomc_pivot_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        # fomc_sentiment 已为 T+1 生效的连续前向填充阶梯数据
        sentiment = data['fomc_sentiment'].ffill()
        
        # 【边际变化铁律】: 绝对禁止直接看 absolute level，使用 .diff(5) 捕捉阶梯的瞬间跳变
        # 因为在阶梯跳变后的第 1 到 5 天，.diff(5) 会保持这个跳变值，巧妙地形成了 5 个交易日的事件窗口脉冲
        diff_5 = sentiment.diff(5)
        prev_sentiment = sentiment.shift(5)
        
        signal = pd.Series(0.0, index=data.index)
        
        # ==========================================
        # 多头脉冲 (Bull Pulse): 政策转向鸽派宽松，注入流动性预期
        # ==========================================
        # 1. 绝对突跃: 情绪得分短期内急剧上升超过整个分值域([-1, 1])的 1/8 (0.25)
        is_dovish_surge = diff_5 > 0.25
        # 2. 鹰鸽反转: 前期态度为鹰(负)，现期为鸽(正)，且跨越零轴的跳跃幅度 > 0.15
        is_dovish_turn = (sentiment > 0.0) & (prev_sentiment < 0.0) & (diff_5 > 0.15)
        
        bull_cond = is_dovish_surge | is_dovish_turn
        
        # ==========================================
        # 空头脉冲 (Bear Pulse): 政策转向鹰派收紧，压制风险资产流动性
        # ==========================================
        # 1. 绝对骤降: 情绪得分急剧恶化下降超 -0.25
        is_hawkish_surge = diff_5 < -0.25
        # 2. 鸽鹰反转: 前期态度为鸽(正)，现期为鹰(负)，且跳跃幅度 < -0.15
        is_hawkish_turn = (sentiment < 0.0) & (prev_sentiment > 0.0) & (diff_5 < -0.15)
        
        bear_cond = is_hawkish_surge | is_hawkish_turn
        
        # 赋值狙击手级别的事件脉冲 (-1.0 或 1.0)
        signal.loc[bull_cond] = 1.0
        signal.loc[bear_cond] = -1.0
        
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"