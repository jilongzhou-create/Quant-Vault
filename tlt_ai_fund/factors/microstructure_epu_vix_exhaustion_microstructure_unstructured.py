import numpy as np
import pandas as pd

class MicrostructureEpuVixExhaustionFactor:
    """微观恐慌与文本不确定性突变衰竭因子 (microstructure/unstructured)

    逻辑: 结合期权微观流动性恐慌(VIX)与非结构化文本的政策恐慌突变(EPU 5日动量)。严格遵循二阶导反接飞刀原则：当政策不确定性在短期内剧烈飙升且微观情绪处于绝对极值(VIX Z-Score > 2.5)时，处于抛售主跌浪，绝对禁止直接买入美债。只有在恐慌极值确认见顶且日内均值回落(VIX < 3日均值且为负差分)的瞬间，确认系统性抛压耗尽，此时才触发狙击手级脉冲抄底美债(TLT)。
    数据: vixcls (CBOE微观期权恐慌), usepuindxd (基于新闻NLP的经济政策不确定性指数)
    触发: (VIX 252日 Z-Score > 2.5 且 EPU 5日动量突变 Z-Score > 1.5) 且 (当日VIX < 3日均值 且 VIX.diff() < 0)
    输出: +1.0 脉冲(极端恐慌衰竭时做多TLT)，-1.0 脉冲(极度自满衰竭时做空TLT)，常态严格输出 0.0
    """

    def __init__(self):
        self.name = 'microstructure_epu_vix_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，狙击手脉冲，默认全0
        signal = pd.Series(0.0, index=data.index)

        # 缺失列校验
        required_cols = ['vixcls', 'usepuindxd']
        if not all(col in data.columns for col in required_cols):
            return signal

        vix = data['vixcls'].ffill()
        epu = data['usepuindxd'].ffill()

        # ---------------------------------------------------------
        # 指标1：VIX 微观恐慌绝对极值判定 (252日滚动 Z-Score)
        # ---------------------------------------------------------
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std().replace(0.0, np.nan)
        vix_zscore = (vix - vix_mean) / vix_std

        # ---------------------------------------------------------
        # 指标2：EPU 文本指数边际变化突变判定 (遵守铁律3：边际变化)
        # ---------------------------------------------------------
        # 绝对禁止使用水平值，计算5日波动动量，捕捉预期快速恶化的突变瞬间
        epu_diff5 = epu.diff(5)
        epu_diff5_mean = epu_diff5.rolling(window=252, min_periods=60).mean()
        epu_diff5_std = epu_diff5.rolling(window=252, min_periods=60).std().replace(0.0, np.nan)
        epu_diff5_zscore = (epu_diff5 - epu_diff5_mean) / epu_diff5_std

        # ---------------------------------------------------------
        # 指标3：反接飞刀衰竭条件 (遵守铁律2：二阶导数)
        # ---------------------------------------------------------
        vix_3d_ma = vix.rolling(window=3, min_periods=1).mean()
        vix_diff = vix.diff()
        
        # 恐慌衰竭：VIX 跌破过去3日均值，且边际日内动量转负 (确认不再创新高)
        panic_exhausting = (vix < vix_3d_ma) & (vix_diff < 0)
        
        # 自满衰竭：VIX 向上突破3日均值，且边际日内动量转正
        complacency_exhausting = (vix > vix_3d_ma) & (vix_diff > 0)

        # ---------------------------------------------------------
        # 触发逻辑：共振组合产生脉冲
        # ---------------------------------------------------------
        # 多头触发脉冲：微观绝对极值 + 非结构化文本恐慌激增 + 恐慌衰竭确认
        long_cond = (vix_zscore > 2.5) & (epu_diff5_zscore > 1.5) & panic_exhausting

        # 空头触发脉冲：微观极度自满无压力 + 文本情绪极度平稳 + 波动率开始抬头
        short_cond = (vix_zscore < -1.5) & (epu_diff5_zscore < -1.5) & complacency_exhausting

        # 仅在触发时生成脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"