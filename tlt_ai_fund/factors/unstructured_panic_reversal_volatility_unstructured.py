import numpy as np
import pandas as pd

class UnstructuredPanicReversalFactor:
    """非结构化恐慌与情绪反转脉冲因子 (volatility/unstructured)

    逻辑: 整合两类非结构化文本数据挖掘脉冲反转。
          1. EPU(经济政策不确定性新闻指数): 捕捉极端政策恐慌。当政策不确定性极度狂飙并开始衰竭时，意味着宏观冲击被市场充分定价，避险资金回流美债。
          2. FOMC情绪得分: 捕捉央行语调边际突变。摒弃绝对水位，仅当情绪发生2.5倍标准差级别的边际跳跃反转时，触发顺势脉冲。
    数据: usepuindxd (经济政策不确定性), vixcls (VIX跨资产确认), fomc_sentiment (FOMC情绪得分)
    触发: 
      - 条件A: EPU 252日 Z-Score > 2.5 且开始回落 (diff < 0) + VIX同步回落 -> 恐慌衰竭看多脉冲 (+1.0)
      - 条件B: FOMC情绪 5日变化量 > 2.5σ 且完成由负转正(鹰转鸽) -> 边际突变看多脉冲 (+1.0)
    输出: [-1.0, 1.0] 的严格狙击手级离散脉冲信号，非触发日常态休眠为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_panic_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态信号必须严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        has_epu = 'usepuindxd' in data.columns
        has_vix = 'vixcls' in data.columns
        has_fomc = 'fomc_sentiment' in data.columns
        
        # --- 逻辑1: 经济政策不确定性(EPU)极值与跨资产衰竭 ---
        if has_epu and has_vix:
            epu = data['usepuindxd'].ffill()
            vix = data['vixcls'].ffill()
            
            # EPU新闻指数日频噪音极大，使用1周(5日)移动平均提取核心趋势水位
            epu_smooth = epu.rolling(window=5).mean()
            
            # 衡量宏观政策恐慌水位的 252日 Z-Score (识别极端拥挤)
            epu_z = (epu_smooth - epu_smooth.rolling(window=252).mean()) / epu_smooth.rolling(window=252).std()
            
            # 铁律2: 二阶导数，绝对禁止直接买入。必须极值出现后，指标跌破3日均值且边际变化为负
            epu_reversal_down = (epu_smooth < epu_smooth.rolling(window=3).mean()) & (epu_smooth.diff() < 0)
            vix_reversal_down = (vix < vix.rolling(window=3).mean()) & (vix.diff() < 0)
            
            epu_reversal_up = (epu_smooth > epu_smooth.rolling(window=3).mean()) & (epu_smooth.diff() > 0)
            vix_reversal_up = (vix > vix.rolling(window=3).mean()) & (vix.diff() > 0)
            
            # 极端不确定性高点 + EPU衰竭 + VIX同步回落确认 (看多美债)
            long_epu = (epu_z > 2.5) & epu_reversal_down & vix_reversal_down
            # 极端自满低点(-2.0σ) + EPU风险发酵 + VIX同步上行确认 (看空美债)
            short_epu = (epu_z < -2.0) & epu_reversal_up & vix_reversal_up
            
            signal[long_epu] = 1.0
            signal[short_epu] = -1.0

        # --- 逻辑2: 央行政策情绪(FOMC)边际突变 ---
        if has_fomc:
            fomc = data['fomc_sentiment'].ffill()
            
            # 铁律3: 边际变化，绝对禁止直接输出绝对值。使用 5日变化量 捕捉突变瞬间
            fomc_diff5 = fomc.diff(5)
            
            # 使用 252日 滚动标准差衡量突变力度 (设定0.05下限防止平时会议空白期0值导致标准差趋零)
            fomc_std = fomc.diff().rolling(window=252).std().clip(lower=0.05)
            fomc_diff_z = fomc_diff5 / fomc_std
            
            # 预期反转确认: 要求不仅变化量极大，且性质发生质变
            hawk_to_dove = (fomc.shift(5) < 0) & (fomc > 0)
            dove_to_hawk = (fomc.shift(5) > 0) & (fomc < 0)
            
            # 2.5倍标准差级别的突发鸽派惊喜反转 (看多美债)
            long_fomc = (fomc_diff_z > 2.5) & hawk_to_dove
            # 2.5倍标准差级别的突发鹰派惊吓反转 (看空美债)
            short_fomc = (fomc_diff_z < -2.5) & dove_to_hawk
            
            # 脉冲叠加，只要满足任一极端衰竭/突变条件即触发信号
            signal[long_fomc] = 1.0
            signal[short_fomc] = -1.0
            
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"