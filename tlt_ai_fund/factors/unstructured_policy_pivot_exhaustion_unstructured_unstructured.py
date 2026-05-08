import numpy as np
import pandas as pd

class UnstructuredPolicyPivotExhaustionFactor:
    """Unstructured 政策预期突变与不确定性动量衰竭因子

    逻辑: 结合了新闻文本情绪(usepuindxd)与央行会议情绪(fomc_sentiment)的双重非结构化脉冲因子。
          当经济政策不确定性激增至极端恐慌并开始边际衰竭时，市场从流动性挤兑进入纯避险模式，看多美债；
          当FOMC鸽派情绪边际跃升且动量见顶时，确认超预期放水已被初步定价，随后发酵看多美债。
          相反，过度安逸破灭或超预期紧缩时，脉冲看空。
    数据: usepuindxd (美国经济政策不确定性指数), fomc_sentiment (FOMC鸽鹰情绪得分)
    触发: 一阶导数(动量) Z-Score 达极端分位 (2.0/2.5) + 二阶导数(反转) < 0。
    输出: +1.0 (鸽派/避险脉冲，看多TLT) / -1.0 (鹰派/风险偏好复苏，看空TLT)
    """

    def __init__(self):
        self.name = 'unstructured_policy_pivot_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        has_epu = 'usepuindxd' in data.columns
        has_fomc = 'fomc_sentiment' in data.columns

        if not has_epu and not has_fomc:
            return signal

        epu_long = pd.Series(False, index=data.index)
        epu_short = pd.Series(False, index=data.index)
        fomc_long = pd.Series(False, index=data.index)
        fomc_short = pd.Series(False, index=data.index)

        # ---------------------------------------------------------------------
        # 1. 新闻文本政策不确定性衰竭逻辑 (EPU)
        # ---------------------------------------------------------------------
        if has_epu:
            # 避免噪音，提取周度趋势中枢
            epu = data['usepuindxd'].ffill()
            epu_trend = epu.rolling(5).mean()
            
            # 边际变化：计算两周时间窗的加速动量
            epu_momentum = epu_trend.diff(10)
            
            # 动量极值测度
            epu_zscore = (epu_momentum - epu_momentum.rolling(252).mean()) / (epu_momentum.rolling(252).std() + 1e-6)
            
            # 二阶导数：动量开始反转(衰竭条件，防接飞刀)
            epu_exhaustion_up = epu_momentum.diff(1) < 0
            epu_exhaustion_down = epu_momentum.diff(1) > 0
            
            # 触发脉冲: EPU飙升极值+确认衰竭 -> 资金恐慌抛售结束，转向长久期无风险美债避险 (+1.0)
            epu_long = (epu_zscore > 2.0) & epu_exhaustion_up
            # 触发脉冲: EPU跌落低谷+确认回升 -> 极端安逸破灭，紧缩预期酝酿，利空长债 (-1.0)
            epu_short = (epu_zscore < -2.0) & epu_exhaustion_down

        # ---------------------------------------------------------------------
        # 2. 会议文本央行情绪突变衰竭逻辑 (FOMC)
        # ---------------------------------------------------------------------
        if has_fomc:
            fomc = data['fomc_sentiment'].ffill()
            
            # 利用指数平滑转化低频阶梯数据为连续势能，然后计算边际跳跃
            # 严格遵守"必须用变化量捕捉预期跳跃"的铁律
            fomc_momentum = fomc.ewm(span=5).mean().diff(3)
            
            # 动量极值测度
            fomc_zscore = (fomc_momentum - fomc_momentum.rolling(252).mean()) / (fomc_momentum.rolling(252).std() + 1e-6)
            
            # 二阶导数：突发预期发酵的冲顶衰竭
            fomc_exhaustion_dovish = fomc_momentum.diff(1) < 0
            fomc_exhaustion_hawkish = fomc_momentum.diff(1) > 0
            
            # 触发脉冲: 超预期转鸽脉冲极值+确认拐点 -> 长端利率下行定价 (+1.0)
            fomc_long = (fomc_zscore > 2.5) & fomc_exhaustion_dovish
            # 触发脉冲: 超预期转鹰脉冲极值+确认拐点 -> 长端利率飙升定价 (-1.0)
            fomc_short = (fomc_zscore < -2.5) & fomc_exhaustion_hawkish

        # ---------------------------------------------------------------------
        # 3. 脉冲信号合成 (零值休眠)
        # ---------------------------------------------------------------------
        signal[epu_long | fomc_long] = 1.0
        signal[epu_short | fomc_short] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"