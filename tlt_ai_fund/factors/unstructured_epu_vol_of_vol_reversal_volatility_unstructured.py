import numpy as np
import pandas as pd

class UnstructuredEpuVolOfVolReversalFactor:
    """非结构化政策不确定性波动率反转脉冲因子 (volatility/unstructured)

    逻辑: 衡量基于新闻文本的经济政策不确定性(EPU)本身的波动率(Vol of Vol)。当非结构化的政策预期极其混乱、波动率飙升至极值并开始衰竭时，意味着去杠杆恐慌和不确定性溢价见顶，美债回归避险和正向Carry属性，触发看多脉冲；反之，当情绪极度自满(波动率跌至冰点)且突发边际扰动飙升时，提示风险重估，触发看空脉冲。因子严格输出脉冲信号，常态休眠。
    数据: usepuindxd (经济政策不确定性指数 - 纯非结构化文本代理数据)
    触发: 多头(+1.0): 252日 Vol Z-Score > 2.5 (极值) 且 Vol.diff(3) < 0 且 EPU.diff(3) < 0 (波动与情绪双重衰竭确认)。
          空头(-1.0): 252日 Vol Z-Score < -2.0 (拥挤死水) 且 Vol.diff(1) > 0 且 EPU.diff(1) > 0 (边际冲击突现)。
    输出: [-1.0, 1.0] 的离散脉冲信号。
    """

    def __init__(self):
        self.name = 'unstructured_epu_vol_of_vol_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 初始信号全为0.0 (零值休眠)
        signal = pd.Series(0.0, index=data.index)

        # 防御性检查: 如果所需数据不在输入中，直接返回全 0
        if 'usepuindxd' not in data.columns:
            return signal

        # 数据清洗: 填补缺失值防前瞻
        epu = data['usepuindxd'].ffill()

        # 衍生计算: 非结构化波动的动量特征
        # 21日为经济学标准的单月交易日，衡量短期政策预期的混乱程度
        epu_vol = epu.rolling(window=21).std()

        # 252日为经济学标准年度交易日，构建绝对水位锚定
        epu_vol_mean = epu_vol.rolling(window=252).mean()
        epu_vol_std = epu_vol.rolling(window=252).std()
        
        # 防止除 0 导致的数据无限大错误
        epu_vol_std = epu_vol_std.replace(0.0, np.nan)
        epu_vol_zscore = (epu_vol - epu_vol_mean) / epu_vol_std

        # 铁律3: 边际变化 (使用 1日和 3日 diff 捕捉变化的发生瞬间)
        epu_vol_diff_3d = epu_vol.diff(3)
        epu_diff_3d = epu.diff(3)
        epu_vol_diff_1d = epu_vol.diff(1)
        epu_diff_1d = epu.diff(1)

        # 铁律2: 二阶导数 (极值 + 衰竭 = 拒绝接飞刀)
        # 多头条件: 波动率年度 Z-Score > 2.5 的极端狂飙 + 波动率及绝对值见顶回落
        long_condition = (
            (epu_vol_zscore > 2.5) & 
            (epu_vol_diff_3d < 0) & 
            (epu_diff_3d < 0)
        )

        # 空头条件: 波动率年度极度低迷死水 (Z < -2.0) + 波动率及绝对值突发向上脉冲
        short_condition = (
            (epu_vol_zscore < -2.0) & 
            (epu_vol_diff_1d > 0) & 
            (epu_diff_1d > 0)
        )

        # 严格执行脉冲赋值，非触发日保持默认的 0.0
        signal.loc[long_condition] = 1.0
        signal.loc[short_condition] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"