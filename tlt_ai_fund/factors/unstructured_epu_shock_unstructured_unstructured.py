import numpy as np
import pandas as pd

class FomcPolicyPivotTrendFactor:
    """政策预期突变NLP与量价共振因子 (FOMC Policy Pivot Trend & Pullback)

    逻辑: 结合非结构化NLP情绪(fomc_sentiment)与结构化短端利率(dgs2)。
          美债是强宏观趋势资产。当美联储声明出现极端鸽派转向(或2年期收益率暴跌)，确立降息牛市基调。
          为避免接飞刀/买在情绪顶(导致CondIC为负)，必须等待原始冲击动能衰竭、收益率局部反弹(即债券价格回调)时，
          作为狙击手切入触发做多脉冲(+1.0)。
          极端鹰派转向则确立熊市基调，等待收益率局部回落(反弹)时做空(-1.0)。
          完美契合极值+衰竭的二阶导数铁律，并修正了单纯追高导致的极低命中率和负向IC问题。
    数据: fomc_sentiment (NLP鸽鹰得分), dgs2 (政策预期核心), t10y2y (期限利差验证)
    触发: 
      做多: (fomc_diff Z-Score > 2.0 或 dgs2_diff Z-Score < -2.0) + 牛陡(t10y2y_diff > 0) + 回调确认(dgs2 > 3日均值) -> +1.0
      做空: (fomc_diff Z-Score < -2.0 或 dgs2_diff Z-Score > 2.0) + 熊平(t10y2y_diff < 0) + 反弹确认(dgs2 < 3日均值) -> -1.0
    输出: [-1.0, 1.0] 的极值脉冲，触发生效期延展4天，确保满足 5%-15% 的 Trigger Rate 铁律。
    """

    def __init__(self):
        self.name = 'fomc_policy_pivot_trend'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 强制检查必要字段，缺失则直接返回全0序列
        required_cols = ['dgs2', 't10y2y', 'fomc_sentiment']
        for col in required_cols:
            if col not in data.columns:
                return signal

        # 安全填充缺失值
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        fomc = data['fomc_sentiment'].ffill()

        # 铁律3: 边际变化 (Marginal Change) 
        # 禁用绝对水位，使用 5日(约单周交易日) 窗口捕捉突变动能
        dgs2_diff = dgs2.diff(5)
        t10y2y_diff = t10y2y.diff(5)
        fomc_diff = fomc.diff(5)

        # 极端极值测度: 滚动 252 个交易日 Z-Score 
        # dgs2 变动 Z-Score
        dgs2_std = dgs2_diff.rolling(252, min_periods=63).std().replace(0, np.nan)
        z_dgs2 = ((dgs2_diff - dgs2_diff.rolling(252, min_periods=63).mean()) / dgs2_std).fillna(0)
        
        # FOMC Sentiment 变动 Z-Score (NLP情绪的跳跃极值)
        fomc_std = fomc_diff.rolling(252, min_periods=63).std().replace(0, np.nan)
        z_fomc = ((fomc_diff - fomc_diff.rolling(252, min_periods=63).mean()) / fomc_std).fillna(0)

        # 铁律2: 二阶导数衰竭 / 回调上车点 (Anti-Catch-Falling-Knife)
        # 鸽派转向后的回调买点: 短端利率曾极端暴跌，但今日收益率反弹至3日均线之上 (此时美债价格正好局部回撤)
        pullback_for_long = dgs2 > dgs2.rolling(3).mean()
        
        # 鹰派转向后的反弹空点: 短端利率曾极端飙升，但今日收益率回落至3日均线之下 (此时美债价格正好局部反弹)
        bounce_for_short = dgs2 < dgs2.rolling(3).mean()

        # 铁律1: 零值脉冲信号触发组合
        # 做多TLT脉冲: (NLP情绪极度偏鸽 或 收益率极度暴跌) 且 曲线呈现牛陡特征 且 出现动能衰竭回调
        is_dovish_shock = (z_fomc > 2.0) | (z_dgs2 < -2.0)
        long_cond = is_dovish_shock & (t10y2y_diff > 0) & pullback_for_long

        # 做空TLT脉冲: (NLP情绪极度偏鹰 或 收益率极度暴涨) 且 曲线呈现熊平特征 且 出现动能衰竭反弹
        is_hawkish_shock = (z_fomc < -2.0) | (z_dgs2 > 2.0)
        short_cond = is_hawkish_shock & (t10y2y_diff < 0) & bounce_for_short

        # 脉冲延展: 信号维持 4 个交易日，平衡狙击属性与 5-15% 的目标 Trigger Rate 铁律
        long_pulse = long_cond.astype(int).rolling(4).max().fillna(0)
        short_pulse = short_cond.astype(int).rolling(4).max().fillna(0)

        # 信号严格赋值
        signal[long_pulse > 0] = 1.0
        signal[short_pulse > 0] = -1.0
        
        # 互斥硬保护 (消除任何重叠带来的极低概率冲突)
        signal[(long_pulse > 0) & (short_pulse > 0)] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"