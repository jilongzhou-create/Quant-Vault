import numpy as np
import pandas as pd

class UnstructuredEpuFomcDivergencePulseFactor:
    """Unstructured EPU & FOMC Divergence Pulse Factor (unstructured/NLP)

    逻辑: 本因子结合两大非结构化NLP数据(每日经济政策不确定性指数 EPU 与美联储声明情绪得分 FOMC Sentiment)。
          当宏观政策不确定性极度飙升(EPU处于历史极值)引发市场恐慌时，若美联储终于边际转鸽(FOMC情绪突发鸽派跳跃)，将打破恐慌螺旋，引发美债(TLT)主升浪。
          为遵守反接飞刀铁律，必须等待 EPU 动能衰竭(拐头向下)与 Fed 边际突变的共振瞬间，方可触发做多美债的狙击手脉冲信号。反之亦然。
    数据: usepuindxd (日常新闻政策不确定性), fomc_sentiment (央行文本鹰鸽情绪)
    触发: 
      极值: EPU 252日 Z-Score > 2.5 (不确定性极高) 或 < -2.0 (极度自满)
      衰竭 (二阶导数): EPU 3日变化量 < 0 (极度恐慌见顶回落) 或 > 0 (自满见底反弹)
      边际变化: FOMC情绪近5日变化量的 Z-Score > 2.0 (鸽派突变) 或 < -2.0 (鹰派突变)
    输出: 仅在共振发生的首日输出 +1.0 (做多) 或 -1.0 (做空) 的单日脉冲，常态休眠为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_epu_fomc_divergence_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1：零值休眠，默认全 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必要数据字段
        if 'usepuindxd' not in data.columns or 'fomc_sentiment' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        fomc = data['fomc_sentiment'].ffill()

        # =================================================================
        # 条件组1 & 2: EPU 极值状态与衰竭条件 (反接飞刀铁律)
        # =================================================================
        # 计算 EPU 的 252 日标准 Z-Score
        epu_mean = epu.rolling(window=252, min_periods=21).mean()
        epu_std = epu.rolling(window=252, min_periods=21).std().clip(lower=1e-5)
        epu_zscore = (epu - epu_mean) / epu_std
        
        # 二阶导数/衰竭：恐慌极值必须伴随动能回落 (3日差分 < 0) 方可抄底
        epu_panic_exhaustion = epu.diff(3) < 0
        # 自满极值必须伴随动能反弹 (3日差分 > 0) 方可做空
        epu_complacency_exhaustion = epu.diff(3) > 0

        # =================================================================
        # 条件组3: FOMC 情绪边际突变 (边际变化铁律)
        # =================================================================
        # 绝对禁止使用绝对值，使用 5 日变化量捕捉每次 FOMC 声明的突变跳跃
        fomc_diff = fomc.diff(5)
        fomc_diff_mean = fomc_diff.rolling(window=252, min_periods=21).mean()
        fomc_diff_std = fomc_diff.rolling(window=252, min_periods=21).std().clip(lower=1e-5)
        fomc_diff_z = (fomc_diff - fomc_diff_mean) / fomc_diff_std

        # 鸽派突变：边际变化为正(转鸽) 且幅度达历史极端 (Z > 2.0)
        fomc_dovish_shock = (fomc_diff > 0) & (fomc_diff_z > 2.0)
        # 鹰派突变：边际变化为负(转鹰) 且幅度达历史极端 (Z < -2.0)
        fomc_hawkish_shock = (fomc_diff < 0) & (fomc_diff_z < -2.0)

        # =================================================================
        # 信号合成与脉冲控制
        # =================================================================
        # 看多脉冲：EPU极度恐慌 + 恐慌开始衰竭 + 央行恰好释放明确鸽派转向信号 = 做多美债
        long_condition = (epu_zscore > 2.5) & epu_panic_exhaustion & fomc_dovish_shock
        
        # 看空脉冲：EPU极度自满 + 避险情绪抬头 + 央行恰好释放鹰派紧缩信号 = 做空美债
        short_condition = (epu_zscore < -2.0) & epu_complacency_exhaustion & fomc_hawkish_shock

        # 严格限制为"狙击手脉冲"：仅在条件首次满足的当天触发，滤除连续的 +1.0/-1.0
        is_first_long_day = long_condition & (~long_condition.shift(1).fillna(False))
        is_first_short_day = short_condition & (~short_condition.shift(1).fillna(False))

        # 赋值并命名
        signal.loc[is_first_long_day] = 1.0
        signal.loc[is_first_short_day] = -1.0
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"