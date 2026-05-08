import numpy as np
import pandas as pd

class UnstructuredOptionsCrossVolExhaustionFactor:
    """Unstructured Options Cross Vol Exhaustion (unstructured/options)

    逻辑: 通过交叉对比黄金隐含波动率(gvzcls)、股市隐含波动率(vixcls)和经济政策不确定性(usepuindxd)，
          精准捕捉两种截然不同的宏观恐慌衰竭脉冲：
          1. 通胀/政策恐慌衰竭: 当黄金波动率或政策不确定性处于极值(Z>1.2)且开始回落时，表明紧缩/通胀恐慌见顶，美联储极大概率转鸽，美债迎来强烈的修复性做多脉冲(+1.0)。
          2. 纯增长恐慌衰竭: 当仅股市波动率处于极值(Z>1.2)，而黄金波动率和政策不确定性正常，且VIX开始回落时，表明纯粹的衰退恐慌解除，资金重新恢复风险偏好(Risk-On)，流出避险美债(-1.0)。
    数据: usepuindxd (经济政策不确定性), vixcls (美股恐慌), gvzcls (黄金恐慌)
    触发: (gvz或epu Z>1.2 & 双回落) -> +1.0; (vix Z>1.2 & gvz,epu<1.0 & vix回落) -> -1.0
    输出: 严格脉冲信号，极端恐慌衰竭瞬间触发 +1.0 或 -1.0，其余状态强制休眠为 0.0
    """

    def __init__(self):
        self.name = 'unstructured_options_crossvol_exhaustion'
        self.window = 252

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号必须严格设为 0.0 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)

        # 检查所需数据字段是否存在，缺失则直接返回全0信号
        required_cols = ['usepuindxd', 'vixcls', 'gvzcls']
        for col in required_cols:
            if col not in data.columns:
                return signal

        # 处理缺失值 (前向填充以确保日频序列连续性)
        df = data[required_cols].ffill()

        # 计算 252日滚动 Z-Score (使用 min_periods=60 保证初期有信号)
        # 这里捕捉"指标处于极端高位"的第一条件
        vix_mean = df['vixcls'].rolling(self.window, min_periods=60).mean()
        vix_std = df['vixcls'].rolling(self.window, min_periods=60).std()
        vix_z = (df['vixcls'] - vix_mean) / vix_std

        gvz_mean = df['gvzcls'].rolling(self.window, min_periods=60).mean()
        gvz_std = df['gvzcls'].rolling(self.window, min_periods=60).std()
        gvz_z = (df['gvzcls'] - gvz_mean) / gvz_std

        epu_mean = df['usepuindxd'].rolling(self.window, min_periods=60).mean()
        epu_std = df['usepuindxd'].rolling(self.window, min_periods=60).std()
        epu_z = (df['usepuindxd'] - epu_mean) / epu_std

        # 计算边际衰竭条件: 当前值跌破3日均值 (二阶导数反接飞刀铁律)
        vix_drop = df['vixcls'] < df['vixcls'].rolling(3).mean()
        gvz_drop = df['gvzcls'] < df['gvzcls'].rolling(3).mean()
        epu_drop = df['usepuindxd'] < df['usepuindxd'].rolling(3).mean()

        # Regime 1: 通胀或政策恐慌见顶衰竭 -> 做多美债 (+1.0)
        # 经济逻辑: 黄金波动率或政策不确定性极高，且两者同步开始回落。说明对加息/恶性通胀的最坏预期已过，美债抄底点出现。
        long_cond = ((gvz_z > 1.2) | (epu_z > 1.2)) & gvz_drop & epu_drop

        # Regime 2: 纯衰退恐慌见顶衰竭 -> 做空美债 (-1.0)
        # 经济逻辑: 股市恐慌极高，但通胀/政策恐慌正常，且股市恐慌开始回落。说明纯风险偏好冲击结束，资金将撤出美债回流股市。
        short_cond = (vix_z > 1.2) & (gvz_z < 1.0) & (epu_z < 1.0) & vix_drop

        # 赋值脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        # 处理因除以零或数据不足导致的 NaN，确保常态绝对为 0.0
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window})"