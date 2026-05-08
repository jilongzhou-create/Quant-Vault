import numpy as np
import pandas as pd

class FomcSentimentReversalFactor:
    """FOMC情绪跨界反转脉冲因子 (unstructured/unstructured)

    逻辑: 捕捉美联储政策声明的极端超预期反转。常态下保持零值休眠；当且仅当FOMC情绪得分发生极端边际跳跃（5日变化量Z-Score极值），且情绪得分发生“鹰鸽跨界反转”（前值与现值异号）时，确认旧预期的彻底衰竭与新趋势的确立，输出短期狙击脉冲。严格防范在情绪连续发酵的主跌浪中接飞刀。
    数据: fomc_sentiment (非结构化央行文本鹰鸽情绪得分)
    触发: 5日变化量 Z-Score > 2.5 且 情绪由负(鹰)转正(鸽) 触发看多脉冲；Z-Score < -2.5 且 由正(鸽)转负(鹰) 触发看空脉冲。仅在边际变化瞬间触发。
    输出: +1.0 (极度鸽派跨界反转看多TLT), -1.0 (极度鹰派跨界反转看空TLT), 常态 0.0
    """

    def __init__(self):
        self.name = 'fomc_sentiment_reversal_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 填补非会议日数据，保持阶梯状连续形态
        fomc = data['fomc_sentiment'].ffill()
        
        # 铁律3: 边际变化 (只关注预期的突变瞬间，禁止使用绝对值直接生成信号)
        diff_1 = fomc.diff(1)
        diff_5 = fomc.diff(5)
        
        # 计算 252日 Z-Score (反映年内相对跳跃的极端程度)
        roll_mean = diff_5.rolling(window=252, min_periods=21).mean()
        roll_std = diff_5.rolling(window=252, min_periods=21).std()
        # 加上 1e-6 避免长时间未开会导致的 0 标准差除零错误
        z_diff_5 = (diff_5 - roll_mean) / (roll_std + 1e-6)
        
        # 铁律2: 二阶导数/衰竭 (必须发生跨界反转，代表旧周期的彻底衰竭与新预期的确立)
        # 前期(5天前)处于鹰派(<0)，当前转为鸽派(>0) -> 鹰派动能衰竭
        reversal_to_dove = (fomc.shift(5) < 0) & (fomc > 0)
        # 前期(5天前)处于鸽派(>0)，当前转为鹰派(<0) -> 鸽派动能衰竭
        reversal_to_hawk = (fomc.shift(5) > 0) & (fomc < 0)
        
        # 铁律1: 初始化全 0 Series (零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        # 脉冲触发条件: 仅在产生变化的当天(diff_1!=0) + 极端边际变化幅度 + 跨界反转
        long_cond = (diff_1 > 0) & (z_diff_5 > 2.5) & reversal_to_dove
        short_cond = (diff_1 < 0) & (z_diff_5 < -2.5) & reversal_to_hawk
        
        # 赋值狙击手脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        # 铁律1延伸: 仅在极端事件当天及随后极短几天内输出非零值 (目标Trigger Rate 5-15%)
        # 使用 ffill(limit=2) 让信号在突变后存续 3 天，防范单日事件脉冲在低频组合中被平滑过滤
        signal = signal.replace(0.0, np.nan).ffill(limit=2).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"