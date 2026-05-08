import numpy as np
import pandas as pd

class UnstructuredMacroPivotPulseFactor:
    """非结构化宏观预期反转脉冲因子 (Unstructured/NLP)

    逻辑: 利用两大非结构化指标捕捉宏观预期反转，通过纯粹的事件驱动与衰竭脉冲避免常态接飞刀。
          1. 政策不确定性(EPU)极度飙升且开始衰竭时，往往预示风险极度释放、避险资金入场；
          2. FOMC央行文本情绪发生超预期的极端跳跃时，反映流动性预期的突然转向。
          只有在这两大类预期极端变化且确认方向的瞬间，因子才输出脉冲信号，其余时间严格休眠。
    数据: usepuindxd (经济政策不确定指数), fomc_sentiment (FOMC文本情绪得分)
    触发: 
      - 看多: (EPU 252日 Z-Score > 2.5 AND EPU向下跌破3日均值 AND 动量<0) OR (FOMC 5日边际跳变 Z-Score > 2.5 AND 当日变化>0)
      - 看空: (EPU 252日 Z-Score < -2.0 AND EPU向上突破3日均值 AND 动量>0) OR (FOMC 5日边际跳变 Z-Score < -2.5 AND 当日变化<0)
    输出: [-1.0, 1.0] 的狙击手级脉冲信号
    """

    def __init__(self):
        self.name = 'unstructured_macro_pivot_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号必须完全休眠
        signal = pd.Series(0.0, index=data.index)
        
        has_epu = 'usepuindxd' in data.columns
        has_fomc = 'fomc_sentiment' in data.columns
        
        if not has_epu and not has_fomc:
            return signal

        long_cond = pd.Series(False, index=data.index)
        short_cond = pd.Series(False, index=data.index)
        
        # 1. 经济政策不确定性 (EPU) 极值衰竭逻辑
        if has_epu:
            epu = data['usepuindxd']
            
            # 计算长周期 Z-Score (锚定 252 个交易日)
            epu_mean = epu.rolling(window=252, min_periods=63).mean()
            epu_std = epu.rolling(window=252, min_periods=63).std().replace(0, np.nan)
            epu_zscore = (epu - epu_mean) / epu_std
            
            # 衰竭/反转条件：利用二阶导数和短均线死叉防止接飞刀
            epu_ma3 = epu.rolling(window=3, min_periods=1).mean()
            epu_diff = epu.diff(1)
            
            # 多头触发：恐慌达到极端值 (>2.5σ) 并且开始确立回落 (下穿均线且边际<0)
            epu_long = (epu_zscore > 2.5) & (epu < epu_ma3) & (epu_diff < 0)
            
            # 空头触发：自满情绪极度扭曲 (<-2.0σ) 并且开始反弹 (上穿均线且边际>0)
            epu_short = (epu_zscore < -2.0) & (epu > epu_ma3) & (epu_diff > 0)
            
            long_cond = long_cond | epu_long
            short_cond = short_cond | epu_short

        # 2. FOMC 声明鹰鸽情绪边际突变逻辑
        if has_fomc:
            fomc = data['fomc_sentiment']
            
            # 铁律要求：严格使用边际变化而不是低频连续绝对值
            # 5 日变动量用于平滑并捕捉阶梯跳变整体幅度
            fomc_diff_5 = fomc.diff(5)
            
            # 计算情绪边际变动的历史分布 Z-Score
            fomc_mean = fomc_diff_5.rolling(window=252, min_periods=63).mean()
            fomc_std = fomc_diff_5.rolling(window=252, min_periods=63).std().replace(0, 1e-6)
            fomc_zscore = (fomc_diff_5 - fomc_mean) / fomc_std
            
            # 零值休眠保证：1日变动识别阶梯信号触发的第一天
            # 借此实现预期落地当日生成孤立脉冲，拒绝平稳期的冗余信号
            fomc_diff_1 = fomc.diff(1)
            
            # 多头触发：文本解析产生极端鸽派转向，并且就在当日爆发
            fomc_long = (fomc_zscore > 2.5) & (fomc_diff_1 > 0)
            
            # 空头触发：文本解析产生极端鹰派转向，并且就在当日爆发
            fomc_short = (fomc_zscore < -2.5) & (fomc_diff_1 < 0)
            
            long_cond = long_cond | fomc_long
            short_cond = short_cond | fomc_short
            
        # 赋值脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"