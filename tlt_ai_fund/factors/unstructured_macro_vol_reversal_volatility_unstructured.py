import numpy as np
import pandas as pd

class UnstructuredMacroVolReversalFactor:
    """Unstructured Macro Volatility & Sentiment Reversal Factor (volatility/unstructured)

    逻辑: 结合非结构化文本情绪(FOMC Sentiment)与宏观不确定性(EPU/VIX/GVZ)的极端反转。
         在政策不确定性或跨资产波动率极度狂飙后衰竭，或美联储情绪发生突变时，输出做多美债的脉冲。
         仅在边际突变和极值衰竭点触发，避免在主跌浪中途接飞刀。
    数据: usepuindxd (经济政策不确定性), vixcls (VIX), gvzcls (黄金波动率), fomc_sentiment (FOMC NLP情绪), t10y2y (收益率曲线)
    触发: 
         多头脉冲 (+1.0): (波动率 252日 Z-Score > 2.5 且开始回落 且 收益率曲线变陡) 或 (FOMC情绪 Z-Score > 2.5 鸽派突变)
         空头脉冲 (-1.0): (波动率 252日 Z-Score < -2.0 且开始飙升 且 收益率曲线变平) 或 (FOMC情绪 Z-Score < -2.5 鹰派突变)
    输出: [-1.0, 1.0] 狙击手级脉冲信号
    """

    def __init__(self):
        self.name = 'unstructured_macro_vol_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0.0 的脉冲信号 (铁律1: 零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        # 依赖字段检查
        required_cols = ['usepuindxd', 'vixcls', 'gvzcls', 'fomc_sentiment', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        df = data[required_cols].ffill()
        
        # =====================================================================
        # 引擎 1: 跨资产与非结构化波动率拥挤反转 (Direction C)
        # =====================================================================
        epu = df['usepuindxd']
        vix = df['vixcls']
        gvz = df['gvzcls']
        t10y2y = df['t10y2y']
        
        # 计算 252个交易日 (1年) 滚动 Z-Score 衡量极值水位
        epu_z = (epu - epu.rolling(252).mean()) / (epu.rolling(252).std() + 1e-6)
        vix_z = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-6)
        gvz_z = (gvz - gvz.rolling(252).mean()) / (gvz.rolling(252).std() + 1e-6)
        
        # 衰竭条件: 二阶导数向下 且 低于3日均线 (铁律2: 防接飞刀)
        epu_exhaust = (epu.diff() < 0) & (epu < epu.rolling(3).mean())
        vix_exhaust = (vix.diff() < 0) & (vix < vix.rolling(3).mean())
        gvz_exhaust = (gvz.diff() < 0) & (gvz < gvz.rolling(3).mean())
        
        # 异动条件: 波动率自极低位开始飙升 (用于识别自满情绪反转做空)
        epu_spike = (epu.diff() > 0) & (epu > epu.rolling(3).mean())
        vix_spike = (vix.diff() > 0) & (vix > vix.rolling(3).mean())
        gvz_spike = (gvz.diff() > 0) & (gvz > gvz.rolling(3).mean())
        
        # 收益率曲线动量确认 (铁律3: 边际变化)
        # diff > 0 意味着 10Y-2Y 走阔 (Bull Steepening 或 Bear Steepening)
        curve_steep = t10y2y.diff() > 0 
        curve_flat = t10y2y.diff() < 0
        
        # 宏观恐慌衰竭 (做多美债)
        # 任意波动率处于极端高位 + VIX回落确认 + 曲线边际变陡确认
        vol_extreme_high = (epu_z > 2.5) | (vix_z > 2.5) | (gvz_z > 2.5)
        vol_exhaustion = vix_exhaust & (epu_exhaust | gvz_exhaust)
        vol_long_pulse = vol_extreme_high & vol_exhaustion & curve_steep
        
        # 极度自满反转 (做空美债)
        # 任意波动率处于极端低位 + VIX开始飙升确认 + 曲线边际变平确认
        vol_extreme_low = (epu_z < -2.0) | (vix_z < -2.0) | (gvz_z < -2.0)
        vol_spiking = vix_spike & (epu_spike | gvz_spike)
        vol_short_pulse = vol_extreme_low & vol_spiking & curve_flat
        
        # =====================================================================
        # 引擎 2: FOMC NLP 情绪边际突变 (Method A)
        # =====================================================================
        fomc = df['fomc_sentiment']
        
        # 铁律3: 绝对禁止使用情绪得分绝对值，必须提取突变瞬间
        fomc_diff = fomc.diff()
        is_fomc_day = (fomc_diff != 0) & (fomc_diff.notna())
        
        # 计算历史真实议息会议(约每年8次)的情绪变化方差，用于规范化当前突变幅度
        fomc_diff_non_zero = fomc_diff.where(is_fomc_day, np.nan)
        rolling_fomc_std = fomc_diff_non_zero.ffill().rolling(10).std()
        
        # 当日跳跃的 Z-Score
        fomc_z = (fomc_diff / (rolling_fomc_std.bfill().ffill() + 1e-6)).fillna(0)
        
        # 5日变化量 Z-Score，用于捕捉阶梯状反转
        fomc_5d_change = fomc.diff(5)
        fomc_5d_z = (fomc_5d_change / (rolling_fomc_std.bfill().ffill() + 1e-6)).fillna(0)
        
        # 鹰转鸽突变脉冲 (做多美债)
        fomc_dovish_pulse = is_fomc_day & (
            (fomc_z > 2.5) | 
            ((fomc_5d_z > 2.5) & (fomc > 0) & (fomc.shift(5) < 0))
        )
        
        # 鸽转鹰突变脉冲 (做空美债)
        fomc_hawkish_pulse = is_fomc_day & (
            (fomc_z < -2.5) | 
            ((fomc_5d_z < -2.5) & (fomc < 0) & (fomc.shift(5) > 0))
        )
        
        # =====================================================================
        # 信号合成
        # =====================================================================
        # 只要有一侧引擎触发极端反转脉冲即刻输出信号
        signal[vol_long_pulse | fomc_dovish_pulse] = 1.0
        signal[vol_short_pulse | fomc_hawkish_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"