import numpy as np
import pandas as pd

class UnstructuredVolExhaustionFactor:
    """Unstructured Volatility Exhaustion Reversal (volatility/unstructured)

    逻辑: 监控非结构化新闻(EPU)、央行情绪(FOMC)和跨资产波动率(VIX/GVZ)的极端脉冲。绝对禁止接飞刀，只有当各维度极端恐慌指标达标(Z-Score>2.5)且确认开始二阶衰竭回落时，才认为流动性抛售或情绪冲击见顶，此时输出脉冲看多美债(TLT)。反之，当过热自满情绪衰竭时看空。
    数据: usepuindxd (经济政策不确定性), gvzcls (黄金波动率), vixcls (VIX), fomc_sentiment (FOMC情感), t10y2y (期限利差)
    触发: 252日 Z-Score > 2.5 且指标回落至 3 日均线下方 (反向相反)。对于低频阶梯数据 FOMC 和利差，严格使用 .diff() 的边际变化 Z-Score 过滤。
    输出: +1.0 (恐慌瓦解/放鸽突发，看多TLT脉冲)，-1.0 (狂热瓦解/放鹰突发，看空TLT脉冲)。其余时间休眠(0.0)。
    """

    def __init__(self):
        self.name = 'unstructured_vol_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['usepuindxd', 'gvzcls', 'vixcls', 'fomc_sentiment', 't10y2y']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return signal
            
        # 前向填充数据，避免 NaN 影响滚动计算
        df = data[required_cols].ffill()
        
        # 内部基础算子
        def calc_z(s: pd.Series, window: int = 252) -> pd.Series:
            return (s - s.rolling(window).mean()) / (s.rolling(window).std() + 1e-6)
            
        def calc_exhaustion(s: pd.Series, window: int = 3) -> pd.Series:
            """二阶导数衰竭铁律：严格要求指标小于短期均线，避免接飞刀"""
            return s < s.rolling(window).mean()
            
        def calc_rebound(s: pd.Series, window: int = 3) -> pd.Series:
            """空头二阶导数确认：低位瓦解开始抬头"""
            return s > s.rolling(window).mean()

        # 核心锚定确认：波动率回落确认
        vix = df['vixcls']
        vix_exhausted = calc_exhaustion(vix)
        vix_rebounding = calc_rebound(vix)
        
        # 1. 政策不确定性脉冲 (基于新闻非结构化数据)
        epu = df['usepuindxd']
        epu_z = calc_z(epu)
        # 极度恐慌且不再加剧 + 大盘波动率同步衰竭
        epu_long = (epu_z > 2.5) & calc_exhaustion(epu) & vix_exhausted
        # 极度自满(无风险意识)瓦解
        epu_short = (epu_z < -2.5) & calc_rebound(epu) & vix_rebounding
        
        # 2. 跨资产避险瓦解脉冲 (硬通货黄金波动率)
        gvz = df['gvzcls']
        gvz_z = calc_z(gvz)
        gvz_long = (gvz_z > 2.5) & calc_exhaustion(gvz) & vix_exhausted
        gvz_short = (gvz_z < -2.5) & calc_rebound(gvz) & vix_rebounding
        
        # 3. 央行非结构化情感突变脉冲 (边际变化铁律)
        fomc = df['fomc_sentiment']
        # 绝对禁止用绝对值，捕捉突发放鸽/放鹰瞬间
        fomc_diff = fomc.diff()
        fomc_z = calc_z(fomc_diff)
        fomc_long = (fomc_z > 2.5) & vix_exhausted
        fomc_short = (fomc_z < -2.5) & vix_rebounding
        
        # 4. 收益率曲线动量突变 (边际变化铁律，捕捉非结构化事件驱动的剧烈陡峭化)
        curve = df['t10y2y']
        curve_mom = curve.diff(3)
        curve_z = calc_z(curve_mom)
        curve_long = (curve_z > 2.5) & calc_exhaustion(curve_mom)
        curve_short = (curve_z < -2.5) & calc_rebound(curve_mom)
        
        # 汇总统配：只要任何一个维度的黑天鹅达到极值且衰竭，即触发狙击脉冲
        long_cond = epu_long | gvz_long | fomc_long | curve_long
        short_cond = epu_short | gvz_short | fomc_short | curve_short
        
        # 零值休眠铁律：初始已全设 0.0，仅在极端状态赋值
        signal[long_cond] = 1.0
        signal[short_cond & (~long_cond)] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"