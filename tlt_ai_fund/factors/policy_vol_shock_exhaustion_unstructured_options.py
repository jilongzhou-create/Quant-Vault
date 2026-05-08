import numpy as np
import pandas as pd

class FomcEpuGvzPulseFactor:
    """政策情绪与期权恐慌共振衰竭因子 (unstructured/options)

    逻辑: 严格限定于非结构化数据(EPU, FOMC情绪)和期权数据(GVZ/VIX)。
          将经济政策不确定性(EPU)与黄金波动率(GVZ)的Z-Score叠加，构建"宏观冲击指数"。
          当冲击达到极值且开始回落(衰竭)时，市场进入方向选择期。单纯的恐慌衰竭无法分辨利率走向，
          因此必须结合非结构化的 FOMC 情绪得分的边际变化(二阶导)进行定向过滤：
          若美联储边际转鸽，则恐慌衰竭必然带来降息潮，强烈看多美债；
          若美联储边际转鹰，则说明是通胀型恐慌(如2022年)，美联储会硬扛冲击继续紧缩，强烈看空美债。
          常态下因子严格输出0，仅在极端情绪边际突变时输出脉冲。
    数据: usepuindxd (Unstructured), fomc_sentiment (Unstructured), gvzcls/vixcls (Options)
    触发: 冲击指数极值衰竭 (>1.5且开始回落 或 <-1.5且反弹)，叠加 fomc_sentiment 边际偏离。
    """

    def __init__(self):
        self.name = 'fomc_epu_gvz_pulse_unstructured_options'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，默认全为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 校验所需的核心数据列
        required_cols = ['usepuindxd', 'fomc_sentiment']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        epu = data['usepuindxd'].ffill()
        fomc = data['fomc_sentiment'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change)
        # 禁止直接使用阶梯绝对值，计算 FOMC 情绪相对于过去1个月(21日)均值的动量偏离
        fomc_ma21 = fomc.rolling(21).mean()
        fomc_mom = fomc - fomc_ma21
        
        # 判断政策预期的边际变化方向
        fomc_dovish = fomc_mom > 0.02
        fomc_hawkish = fomc_mom < -0.02
        
        # 构建非结构化维度的 EPU Z-Score (126日 / 半年窗口)
        epu_mean = epu.rolling(126).mean()
        epu_std = epu.rolling(126).std() + 1e-8
        epu_z = (epu - epu_mean) / epu_std
        
        # 构建期权维度的 Volatility Z-Score
        options_z = pd.Series(0.0, index=data.index)
        
        # 优先使用 S&P 500 VIX 作为底座
        if 'vixcls' in data.columns:
            vix = data['vixcls'].ffill()
            vix_z = (vix - vix.rolling(126).mean()) / (vix.rolling(126).std() + 1e-8)
            options_z = vix_z
            
        # 如果存在黄金期权波动率(GVZ)，其对避险情绪的表征更好，进行融合覆盖
        if 'gvzcls' in data.columns:
            gvz = data['gvzcls'].ffill()
            gvz_z = (gvz - gvz.rolling(126).mean()) / (gvz.rolling(126).std() + 1e-8)
            options_z = gvz_z.combine_first(options_z)
                
        # 合成跨子域的 "宏观冲击指数" (Unstructured + Options)
        shock_idx = epu_z + options_z
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 用 3 日均值捕获极端情绪的边际拐点
        shock_ma3 = shock_idx.rolling(3).mean()
        
        # 极度恐慌发生，且情绪开始边际回落
        shock_exhausting_down = (shock_idx > 1.5) & (shock_idx < shock_ma3)
        
        # 极度自满(低波动率+低不确定性)发生，且情绪开始反弹破裂
        shock_rebounding_up = (shock_idx < -1.5) & (shock_idx > shock_ma3)
        
        # --- 信号输出逻辑 (狙击手级脉冲) ---
        
        # 场景 A: 极端恐慌衰退
        # 恐慌冲击衰竭 + 美联储边际转鸽 = 通缩/衰退恐慌，降息预期确认 -> 强烈看多美债
        signal[shock_exhausting_down & fomc_dovish] = 1.0
        # 恐慌冲击衰竭 + 美联储边际转鹰 = 通胀型恐慌，联储将继续紧缩 -> 看空美债
        signal[shock_exhausting_down & fomc_hawkish] = -1.0
        
        # 场景 B: 极端自满破裂
        # 自满情绪破裂 + 美联储边际转鹰 = 经济过热引发加息周期 -> 强烈看空美债
        signal[shock_rebounding_up & fomc_hawkish] = -1.0
        # 自满情绪破裂 + 美联储边际转鸽 = 金发姑娘经济高位回落，降息到来 -> 看多美债
        signal[shock_rebounding_up & fomc_dovish] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"