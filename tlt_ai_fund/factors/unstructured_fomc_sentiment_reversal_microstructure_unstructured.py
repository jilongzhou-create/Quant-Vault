import numpy as np
import pandas as pd

class UnstructuredFomcSentimentReversalFactor:
    """FOMC情绪突变与反转脉冲 (挖掘方向: microstructure / 挖掘方法: unstructured)

    逻辑: 捕捉美联储货币政策预期的极端突变与反转。将非结构化文本情绪得分转化为狙击手级脉冲信号。
          采用阶梯数据的边际变化量而非绝对水位，只有当情绪发生极端跳跃（5日预期突变 Z-Score > 2.5），
          且预期方向发生真正的零轴倒转（鹰转鸽/鸽转鹰）时，才确认为反转突发事件并输出脉冲。
    数据: fomc_sentiment (基于 LLM 的 FOMC 鹰鸽情绪得分, 鸽派>0, 鹰派<0)
    触发: 
      - 多头脉冲 (+1.0): 5日动量变化 Z-Score > 2.5 且 情绪水位由负(鹰)转正(鸽)
      - 空头脉冲 (-1.0): 5日动量变化 Z-Score < -2.5 且 情绪水位由正(鸽)转负(鹰)
    输出: 严格脉冲型。触发日及随后极端短期内为 +1.0 / -1.0，非触发常态完全休眠为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_sentiment_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (Sniper Pulse)，初始信号严格设为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # 缺失值前向填充 (保持阶梯状特性，T+1生效逻辑已在底层数据防前瞻)
        fomc = data['fomc_sentiment'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝对禁止直接输出低频数据的绝对值！必须使用 .diff() 捕捉预期突变的瞬间
        # 采用5日动量变化(一周)来度量 FOMC 会议前后的情绪跳跃差值
        fomc_diff5 = fomc.diff(5)
        
        # 计算动量变化的 252 日(一年)滚动 Z-Score，使用 63 日(一季度)作为最小窗口
        roll_mean = fomc_diff5.rolling(window=252, min_periods=63).mean()
        roll_std = fomc_diff5.rolling(window=252, min_periods=63).std().replace(0, np.nan)
        
        # 避免除以0或NaN引发错误
        z_score = (fomc_diff5 - roll_mean) / roll_std
        
        # 铁律2: 二阶导数与衰竭/反转 (Anti-Catch-Falling-Knife)
        # 仅仅变化量大不足以构成买点，必须伴随绝对状态的反转 (预期倒转)
        
        # 条件A: 鹰转鸽反转 (Hawkish to Dovish) -> 看多美债
        # 1. 动量变化处于极端多头突变 (Z-Score > 2.5)
        # 2. 当前情绪已转正 (Dovish)
        # 3. 5天前的情绪为负 (Hawkish)，确保发生了真正的零轴穿越
        cond_dovish = (z_score > 2.5) & (fomc > 0.0) & (fomc.shift(5) < 0.0)
        
        # 条件B: 鸽转鹰反转 (Dovish to Hawkish) -> 看空美债
        # 1. 动量变化处于极端空头突变 (Z-Score < -2.5)
        # 2. 当前情绪已转负 (Hawkish)
        # 3. 5天前的情绪为正 (Dovish)，确保发生了真正的零轴穿越
        cond_hawkish = (z_score < -2.5) & (fomc < 0.0) & (fomc.shift(5) > 0.0)
        
        # 触发脉冲信号
        signal.loc[cond_dovish] = 1.0
        signal.loc[cond_hawkish] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"