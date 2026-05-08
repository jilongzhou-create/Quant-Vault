import numpy as np
import pandas as pd

class UnstructuredSentimentPivotPulseFactor:
    """Unstructured Sentiment Pivot Pulse (unstructured/unstructured)

    逻辑: 结合FOMC声明文本情绪的极值反转与基于新闻的经济政策不确定性(EPU)的恐慌衰竭，捕捉宏观情绪的非线性突变脉冲。当预期发生突然扭转时(如极度鹰派后释放鸽派信号，或极度恐慌后开始平息)，往往是美债长端利率发生拐点的高胜率时刻。
    数据: fomc_sentiment, usepuindxd
    触发: FOMC: 绝对水位 252日 Z-Score 极值 + diff() 发生反向边际跳跃；EPU: Z-Score > 2.5 极度恐慌 + 跌破3日均线确认衰竭
    输出: +1.0 看多美债(极度鹰派突转鸽派/自满情绪破裂避险升温), -1.0 看空美债(极度鸽派突转鹰派/极度恐慌见顶回落), 常态为 0.0
    """

    def __init__(self):
        self.name = 'unstructured_sentiment_pivot_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        has_fomc = 'fomc_sentiment' in data.columns
        has_epu = 'usepuindxd' in data.columns

        if not has_fomc and not has_epu:
            return signal

        long_pulse = pd.Series(False, index=data.index)
        short_pulse = pd.Series(False, index=data.index)

        # 1. FOMC NLP Sentiment Logic (低频阶梯数据，必须使用二阶导和边际变化)
        if has_fomc:
            fomc = data['fomc_sentiment'].ffill()
            
            # 使用1年(252日)滚动窗口计算情绪的绝对水位极值
            fomc_mean = fomc.rolling(window=252, min_periods=21).mean()
            fomc_std = fomc.rolling(window=252, min_periods=21).std()
            fomc_z = (fomc - fomc_mean) / fomc_std.replace(0, 1e-5)
            
            # 计算当天的边际变化，捕捉阶梯跳跃的瞬间
            fomc_diff = fomc.diff(1)
            
            # 看多美债(Dovish Pivot): 原本处于极端鹰派周期 (Z < -2.0)，今日情绪边际转鸽 (diff > 0)
            fomc_long = (fomc_z.shift(1) < -2.0) & (fomc_diff > 0)
            
            # 看空美债(Hawkish Pivot): 原本处于极端鸽派周期 (Z > 2.0)，今日情绪边际转鹰 (diff < 0)
            fomc_short = (fomc_z.shift(1) > 2.0) & (fomc_diff < 0)
            
            # 脉冲信号向后延展3天，以覆盖事件冲击的交易窗口，并使触发率达标(Target: 5%-15%)
            long_pulse = long_pulse | fomc_long.rolling(3).max().fillna(0).astype(bool)
            short_pulse = short_pulse | fomc_short.rolling(3).max().fillna(0).astype(bool)

        # 2. EPU Economic Policy Uncertainty Logic (连续数据，必须使用"极值+均线衰竭")
        if has_epu:
            epu = data['usepuindxd'].ffill()
            
            epu_mean = epu.rolling(window=252, min_periods=21).mean()
            epu_std = epu.rolling(window=252, min_periods=21).std()
            epu_z = (epu - epu_mean) / epu_std.replace(0, 1e-5)
            
            # 3日均线作为短期趋势的衰竭判定线
            epu_ma3 = epu.rolling(window=3, min_periods=1).mean()
            
            # 看多美债(Risk-Off Pulse): 极端自满 (Z < -2.0) 被打破，不确定性突然上升向上突破均线
            epu_long = (epu_z.shift(1) < -2.0) & (epu > epu_ma3)
            
            # 看空美债(Risk-On Pulse): 极端恐慌 (Z > 2.5) 开始衰竭，不确定性向下回落跌破均线 (禁止接飞刀)
            epu_short = (epu_z.shift(1) > 2.5) & (epu < epu_ma3)
            
            # 脉冲信号向后延展2天
            long_pulse = long_pulse | epu_long.rolling(2).max().fillna(0).astype(bool)
            short_pulse = short_pulse | epu_short.rolling(2).max().fillna(0).astype(bool)

        # 赋值并处理多空冲突
        signal.loc[long_pulse] = 1.0
        signal.loc[short_pulse] = -1.0
        
        # 如果当天同一时间既触发了Long又触发了Short，说明宏观信号矛盾，强制归零
        conflict = long_pulse & short_pulse
        signal.loc[conflict] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"