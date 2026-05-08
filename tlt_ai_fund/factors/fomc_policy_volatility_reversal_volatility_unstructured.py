import numpy as np
import pandas as pd

class FomcPolicyVolatilityReversalFactor:
    """FOMC情绪与政策波动率共振反转因子 (volatility/unstructured)

    逻辑: 将非结构化文本数据(FOMC情绪与新闻政策不确定性EPU)与市场波动率(VIX)结合。
          在极度恐慌(EPU或VIX Z-Score>2.5)且恐慌指标开始二阶衰竭(回落)时，
          若FOMC并未持续释放鹰派信号，说明抛压枯竭且无宏观紧缩的"飞刀"，触发做多美债脉冲。
          此外，若FOMC出现超预期鸽派突变(动量变化>2σ)且恐慌消退，强制触发做多。
          极度自满且波动率抬头、或鹰派突变时触发做空脉冲。完美契合零值休眠与二阶衰竭铁律。
    数据: usepuindxd (经济政策不确定性), vixcls (VIX), fomc_sentiment (FOMC情绪得分)
    触发: (极端恐慌 + 衰竭 + 非鹰派确认) 或 (鸽派突变 + 恐慌消退) -> +1.0
    输出: 脉冲型信号 [-1.0, 1.0]，正值看多美债(TLT)
    """

    def __init__(self):
        self.name = 'fomc_policy_vol_reversal_unstruct'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，默认全为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖列
        required_cols = ['usepuindxd', 'vixcls', 'fomc_sentiment']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        fomc = data['fomc_sentiment'].ffill()
        
        # 1. 绝对水位极值：计算 252日滚动 Z-Score
        epu_mean = epu.rolling(252, min_periods=63).mean()
        epu_std = epu.rolling(252, min_periods=63).std()
        epu_zscore = (epu - epu_mean) / epu_std
        
        vix_mean = vix.rolling(252, min_periods=63).mean()
        vix_std = vix.rolling(252, min_periods=63).std()
        vix_zscore = (vix - vix_mean) / vix_std
        
        # 2. 铁律2: 二阶导数与衰竭条件 (Anti-Catch-Falling-Knife)
        epu_diff = epu.diff(1)
        vix_diff = vix.diff(1)
        epu_ma3 = epu.rolling(3).mean()
        vix_ma3 = vix.rolling(3).mean()
        
        # 衰竭判定：严格要求至少一个核心指标出现实质性回落(动量<0且下破3日均线)，且另一个未在创新高
        epu_falling = (epu_diff < 0) & (epu < epu_ma3)
        vix_falling = (vix_diff < 0) & (vix < vix_ma3)
        epu_not_spiking = epu_diff <= 0
        vix_not_spiking = vix_diff <= 0
        
        panic_exhaustion = (epu_falling & vix_not_spiking) | (vix_falling & epu_not_spiking)
        
        # 自满逆转判定：至少一个抬头且上破3日均线，且另一个未在回落
        epu_rising = (epu_diff > 0) & (epu > epu_ma3)
        vix_rising = (vix_diff > 0) & (vix > vix_ma3)
        epu_not_falling = epu_diff >= 0
        vix_not_falling = vix_diff >= 0
        
        complacency_reversal = (epu_rising & vix_not_falling) | (vix_rising & epu_not_falling)
        
        # 3. 铁律3: 边际变化 (Unstructured NLP Shock)
        # 使用5日变化量捕捉FOMC声明发布后的预期跳跃，禁止使用绝对值
        fomc_diff5 = fomc.diff(5)
        fomc_diff_std = fomc_diff5.rolling(252, min_periods=63).std()
        # 计算突变 Z-Score (防止除0)
        fomc_shock_z = fomc_diff5 / (fomc_diff_std + 1e-6)
        
        dovish_shock = (fomc_shock_z > 2.0) & (fomc_diff5 > 0)
        hawkish_shock = (fomc_shock_z < -2.0) & (fomc_diff5 < 0)
        
        # 趋势确认过滤: 防止在持续加息周期接飞刀
        fomc_not_hawkish = fomc_diff5 >= 0
        fomc_not_dovish = fomc_diff5 <= 0
        
        # 4. 狙击手脉冲信号合成
        extreme_panic = (epu_zscore > 2.5) | (vix_zscore > 2.5)
        extreme_complacency = (epu_zscore < -1.5) & (vix_zscore < -1.5)
        
        # 多头触发条件: (极端恐慌 + 开始衰竭 + 央行未再放鹰) 或 (央行鸽派突变 + 恐慌消退确认)
        long_trigger = (extreme_panic & panic_exhaustion & fomc_not_hawkish) | \
                       (dovish_shock & panic_exhaustion)
                       
        # 空头触发条件: (极度自满 + 波动率逆转 + 央行未再放鸽) 或 (央行鹰派突变 + 波动率逆转确认)
        short_trigger = (extreme_complacency & complacency_reversal & fomc_not_dovish) | \
                        (hawkish_shock & complacency_reversal)
        
        # 仅在触发日赋值，非触发日保持默认的 0.0
        signal[long_trigger] = 1.0
        signal[short_trigger] = -1.0
        
        # 剔除由于 rolling/diff 产生的前期 NaN 导致的值
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"