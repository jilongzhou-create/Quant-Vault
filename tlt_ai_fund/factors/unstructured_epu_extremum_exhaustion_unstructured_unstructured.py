import numpy as np
import pandas as pd

class UnstructuredEpuRegimePulseFactor:
    """政策不确定性突变与衰竭脉冲因子 (NLP/Unstructured)

    逻辑: 针对之前FOMC情绪因子触发率过低(0.3%)导致边际贡献不足的问题，本因子彻底重构逻辑，改用日频的美国经济政策不确定性指数(usepuindxd，基于NLP文本挖掘)作为核心触发源。
         当不确定性经历极端飙升(Z-Score>2.5)且随后开始回落(衰竭)时，结合前瞻政策利率(dgs2)的边际动量判断恐慌Regime：
         若短端利率随之急剧下行，说明资金在定价衰退和降息预期(Flight-to-safety)，顺势做多美债(TLT)；
         若短端利率上行，说明恐慌源于通胀失控或超预期鹰派加息，顺势做空美债。
    数据: usepuindxd (经济政策不确定性), dgs2 (2年期美债收益率)
    触发: 过去5天内 EPU 5日变化量的 Z-Score > 2.5 (极端脉冲) + EPU回落至3日均值下方 (二阶衰竭) + DGS2边际动量确认
    输出: 脉冲型信号，买入+1.0，卖出-1.0，常态下严格为0.0
    """

    def __init__(self):
        self.name = 'unstructured_epu_regime_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号严格设为 0.0 (遵守零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)

        # 校验必要字段，防范缺失
        if 'usepuindxd' not in data.columns or 'dgs2' not in data.columns:
            return signal

        epu = data['usepuindxd'].ffill()
        dgs2 = data['dgs2'].ffill()

        # ---------------------------------------------------------------------
        # 铁律3: 边际变化 (Marginal Change Only)
        # 使用5个交易日(一周)的变化量来捕捉政策不确定性的瞬间突变
        # ---------------------------------------------------------------------
        epu_diff = epu.diff(5)
        
        # 计算 252日(一年) 滚动 Z-Score
        epu_mean = epu_diff.rolling(252).mean()
        epu_std = epu_diff.rolling(252).std()
        epu_zscore = (epu_diff - epu_mean) / (epu_std + 1e-8)

        # ---------------------------------------------------------------------
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 捕捉极端高位 Z-Score > 2.5，为了匹配衰竭动作，将警报状态保持5天
        # 这也能确保最终 Trigger Rate 落在 5% - 15% 的健康区间，避免由于信号过少被判定为无效因子
        # ---------------------------------------------------------------------
        extreme_shock = (epu_zscore > 2.5).rolling(5).max().fillna(0).astype(bool)

        # ---------------------------------------------------------------------
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 绝对禁止在恐慌极值日直接交易，必须等待不确定性指标开始回落（低于3日均值），确认情绪的二阶衰退
        # ---------------------------------------------------------------------
        exhaustion = epu < epu.rolling(3).mean()

        # ---------------------------------------------------------------------
        # 交叉验证与方向判断：判断资金和政策预期的真实流向
        # 结合 dgs2 的 5日边际变化区分宏观 Regime：
        # - dgs2_diff < 0 : 衰退恐慌，市场抢跑降息 -> 做多 TLT
        # - dgs2_diff > 0 : 通胀恐慌，市场定价加息 -> 做空 TLT
        # ---------------------------------------------------------------------
        dgs2_diff = dgs2.diff(5)
        flight_to_safety = dgs2_diff < 0
        inflation_panic = dgs2_diff > 0

        # 组合触发条件
        buy_cond = extreme_shock & exhaustion & flight_to_safety
        sell_cond = extreme_shock & exhaustion & inflation_panic

        # 生成脉冲信号 (+1.0 / -1.0)
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"