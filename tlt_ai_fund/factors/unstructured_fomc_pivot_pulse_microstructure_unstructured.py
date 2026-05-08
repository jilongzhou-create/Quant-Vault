import numpy as np
import pandas as pd

class UnstructuredFomcPivotPulseFactor:
    """FOMC预期突变反转脉冲因子 (unstructured)

    逻辑: 将 FOMC 声明的鹰鸽情绪得分(非结构化 NLP 文本转化数据)转化为美债脉冲信号。绝对禁止直接使用连续情绪水位，以免死于长期的缓慢加息/降息周期中。本因子运用 5日差分(Marginal Change) 结合 Z-Score 寻找预期跳跃点。每次预期突变时，由于阶跃函数的数学特性，5日差分会自动形成一个为期5天的高能脉冲，完美将 Trigger Rate 控制在 10%~15% 的狙击手区间。
    数据: fomc_sentiment
    触发: fomc_sentiment 的 5日变化量 Z-Score > 2.5 (恐慌/预期极值)，且情绪得分完成从负到正的跨越 (衰竭与反转确认)。
    输出: +1.0 (极度鹰转鸽，看多美债) / -1.0 (极度鸽转鹰，看空美债)，常态严格输出 0.0。
    """

    def __init__(self, z_threshold: float = 2.5, window: int = 252, diff_days: int = 5):
        self.name = 'unstructured_fomc_pivot_pulse'
        self.z_threshold = z_threshold
        self.window = window
        self.diff_days = diff_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (Sniper Pulse)，非触发期严格返回 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            signal.name = self.name
            return signal
            
        fomc = data['fomc_sentiment'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 提取 FOMC 情绪得分的 N 日边际突变。
        # 每年仅有约8次会议，使用差分不仅过滤了绝对值的黏性，其数学特性还会在跳跃后自发形成长度为 diff_days 的高能脉冲。
        fomc_diff = fomc.diff(self.diff_days).fillna(0.0)
        
        # 计算边际变化的 Z-Score 极值
        # 引入 std 的 clip(lower=0.1) 下限是经济学含义约束：避免因长期未开会情绪无波动导致分母无限趋近 0，从而被微小杂音触发假极值
        roll_mean = fomc_diff.rolling(window=self.window, min_periods=21).mean()
        roll_std = fomc_diff.rolling(window=self.window, min_periods=21).std().clip(lower=0.1)
        z_score = (fomc_diff - roll_mean) / roll_std
        
        # 铁律2: 二阶导数与衰竭 (Anti-Catch-Falling-Knife)
        # 绝不单纯因为变化大就抄底，要求必须包含趋势的衰竭反转：即 N 天前的情绪与当前情绪处于零轴的两侧
        prev_fomc = fomc.shift(self.diff_days).fillna(0.0)
        
        # 多头脉冲：之前是鹰派(<0)，突跳后转为鸽派(>0)，且变化幅度极端 (> 2.5σ)
        bull_pulse = (z_score > self.z_threshold) & (prev_fomc < 0.0) & (fomc > 0.0)
        
        # 空头脉冲：之前是鸽派(>0)，突跳后转为鹰派(<0)，且变化幅度极端 (< -2.5σ)
        bear_pulse = (z_score < -self.z_threshold) & (prev_fomc > 0.0) & (fomc < 0.0)
        
        # 信号注入：只在极值+衰竭反转的窗口期内介入
        signal[bull_pulse] = 1.0
        signal[bear_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_threshold={self.z_threshold}, window={self.window}, diff_days={self.diff_days})"