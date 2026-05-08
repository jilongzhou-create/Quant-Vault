import numpy as np
import pandas as pd

class FomcSentimentPivotPulseFactor:
    """FOMC Sentiment Pivot Pulse (microstructure/unstructured)

    逻辑: 基于美联储声明的NLP非结构化情绪得分。当情绪发生极端跳跃(如鹰派突转鸽派)时, 预期重定价瞬间引发美债脉冲式反弹。为避免接飞刀并满足脉冲铁律, 使用 5 日差分构建动量, 并在动量突破极值(Z > 2.5)且情绪符号完全反转时, 产生为期 5 天的狙击脉冲。此因子的非线性反转逻辑与连续型动量完全正交, 具备极高边际贡献。
    数据: fomc_sentiment (NLP 鹰鸽情绪得分, 范围[-1.0, 1.0], 正值看多美债)
    触发: 5日变化量的 252日 Z-Score > 2.5 且从负转正 -> 鹰转鸽看多脉冲 +1.0; 反之 Z-Score < -2.5 且从正转负 -> 鸽转鹰看空脉冲 -1.0。
    输出: [-1.0, 1.0] 的脉冲信号, 非触发日严格为 0.0。
    """

    def __init__(self):
        self.name = 'fomc_sentiment_pivot_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 初始信号必须为 0.0 (零值休眠)
        signal = pd.Series(0.0, index=data.index)
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        fomc = data['fomc_sentiment'].ffill()
        
        # 铁律3: 边际变化 (只看 5 日动量, 捕捉预期突变, 绝对禁止直接输出绝对值)
        fomc_chg = fomc.diff(5)
        
        # 铁律2: 二阶导数 (极端极值确认)
        # 计算 252 日滚动的 Z-Score, min_periods=21 保证数据初期有信号
        chg_std = fomc_chg.rolling(window=252, min_periods=21).std().replace(0, np.nan)
        chg_mean = fomc_chg.rolling(window=252, min_periods=21).mean()
        fomc_z = (fomc_chg - chg_mean) / chg_std
        
        # 铁律1 & 2 的结合: 零值休眠 (狙击手脉冲) + 反转确认 (从负转正 或 从正转负)
        # 由于 fomc_chg 是 5 日差分, 单次突发事件满足条件后自然会形成一个为期 5 天的极短期信号脉冲
        
        # 鹰转鸽 (Hawkish to Dovish) -> 收益率下行 -> TLT 上涨 (+1.0)
        hawk_to_dove = (fomc_z > 2.5) & (fomc > 0) & (fomc.shift(5) < 0)
        
        # 鸽转鹰 (Dovish to Hawkish) -> 收益率上行 -> TLT 下跌 (-1.0)
        dove_to_hawk = (fomc_z < -2.5) & (fomc < 0) & (fomc.shift(5) > 0)
        
        # 赋值触发信号
        signal[hawk_to_dove] = 1.0
        signal[dove_to_hawk] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"