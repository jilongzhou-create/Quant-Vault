import numpy as np
import pandas as pd

class UnstructuredVolatilityExhaustionFactor:
    """非结构化波动率反转脉冲因子 (volatility/unstructured)

    逻辑: 捕捉非结构化政策预期(EPU/FOMC文本)与跨资产恐慌(VIX/GVZ)极度飙升后的衰竭反转。当政策与市场恐慌双重见顶并开始回落时，意味宏观极度恐慌已被充分计价(Priced-in)，避险情绪修复，产生看多美债脉冲；反之，当极度自满被打破，或FOMC出现边际鹰派跳跃时，产生看空脉冲。常态下严格保持休眠。
    数据: usepuindxd (经济政策不确定性), vixcls (VIX指数), gvzcls (黄金波动率), fomc_sentiment (FOMC文本情绪得分)
    触发: (Z-Score > 2.5 且 动量回落 < 3日均值 且 跨资产确认) 或 (FOMC 情绪边际跳跃 Z-Score > 2.5) -> 触发 +/-1.0
    输出: 狙击手级别的脉冲型信号 [-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'unstructured_vol_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (默认全为 0.0)
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['usepuindxd', 'vixcls', 'gvzcls', 'fomc_sentiment']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 数据预处理: 填充缺失值防止计算中断，防止未来函数
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        fomc = data['fomc_sentiment'].ffill()
        
        # 鲁棒的 Z-Score 计算，引入底噪(min_std)约束，防止长期常数除以零产生的微小扰动放大
        # 252 代表一年交易日，63 代表一个季度
        def calc_zscore(series, min_std):
            mean = series.rolling(252, min_periods=63).mean()
            std = series.rolling(252, min_periods=63).std()
            return (series - mean) / (std + min_std)

        # 1. 极值监控计算 (Extreme Levels)
        # vix 与 gvz 波动率基础底噪设为 0.1；EPU 绝对值大，底噪设为 1.0
        vix_z = calc_zscore(vix, 0.1)
        epu_z = calc_zscore(epu, 1.0)
        gvz_z = calc_zscore(gvz, 0.1)
        
        # 铁律3: 边际变化 (Marginal Change)
        # 对于 FOMC 文本情绪等低频阶梯数据，绝对禁止直接判断绝对水位，必须使用 .diff() 捕捉预期反转跳跃！
        fomc_diff = fomc.diff().fillna(0.0)
        # 设定 0.05 的底噪代表至少需要发生 5% 的政策口吻实质性转移，方可称为“突变”
        fomc_z = calc_zscore(fomc_diff, 0.05) 
        
        # 2. 二阶导数监控计算 (Anti-Catch-Falling-Knife)
        # 恐慌衰竭 (极值见顶并开始回落)
        vix_exhausting = (vix.diff() < 0) & (vix < vix.rolling(3).mean())
        epu_exhausting = (epu.diff() < 0) & (epu < epu.rolling(3).mean())
        gvz_falling = gvz.diff() < 0
        
        # 自满被打破 (极低位突然爆点苏醒)
        vix_waking = (vix.diff() > 0) & (vix > vix.rolling(3).mean())
        epu_waking = (epu.diff() > 0) & (epu > epu.rolling(3).mean())
        gvz_rising = gvz.diff() > 0
        
        # 3. 核心合成逻辑 (条件必须同时满足才能开火)
        
        # 多头触发 (+1.0)
        # A. 恐慌极度拥挤且跨资产确认衰竭 (极值 + 回落 + 黄金恐慌确认回落)
        vol_bull_pulse = ((vix_z > 2.5) & vix_exhausting & gvz_falling) | \
                         ((epu_z > 2.5) & epu_exhausting & gvz_falling)
        # B. NLP 纯粹的 FOMC 边际鸽派突发跳跃 (预期瞬间扭转)
        nlp_bull_pulse = fomc_z > 2.5
        
        # 空头触发 (-1.0)
        # A. 极度自满宁静被打破 (极度负 Z-Score + 苏醒 + 跨资产确认)
        vol_bear_pulse = ((vix_z < -1.5) & vix_waking & gvz_rising) | \
                         ((epu_z < -1.5) & epu_waking & gvz_rising)
        # B. NLP 纯粹的 FOMC 边际鹰派突发跳跃 (加息/紧缩恐慌瞬间跳跃)
        nlp_bear_pulse = fomc_z < -2.5
        
        # 合并脉冲信号
        final_bull = vol_bull_pulse | nlp_bull_pulse
        final_bear = vol_bear_pulse | nlp_bear_pulse
        
        signal.loc[final_bull] = 1.0
        signal.loc[final_bear] = -1.0
        
        # 清除极端情况下同一天发生多空冲突的杂音
        signal.loc[final_bull & final_bear] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(target_asset='TLT', domain='unstructured/volatility')"