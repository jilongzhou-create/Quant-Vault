import numpy as np
import pandas as pd

class UnstructuredPolicyPivotShockFactor:
    """Unstructured Policy Pivot Shock (unstructured/nonlinear)

    逻辑: 捕捉美联储政策预期突变的脉冲冲击。单纯的极端政策突变如果叠加持续攀升的恐慌(EPU)，可能会导致美债遭到无差别抛售(流动性危机)。真正的趋势拐点脉冲(狙击点)必须在：FOMC情绪得分出现极端跳跃，或对政策最敏感的2年期美债(dgs2)出现极端异动并伴随收益率曲线形态(t10y2y)的印证（如降息预期的 Bull Steepening），同时，必须等待经济政策不确定性(EPU)的恐慌极值出现边际回落（衰竭），确认市场消化完毕并开始主升/主跌浪。
    数据: fomc_sentiment, dgs2, t10y2y, usepuindxd
    触发: (FOMC动量Z>2.5 OR dgs2动量Z>2.5且伴随曲线形态确认) AND (EPU Z-Score>2.0且出现均值衰竭)
    输出: +1.0 (鸽派突变看多TLT) / -1.0 (鹰派突变看空TLT) 的极端脉冲信号
    """

    def __init__(self):
        self.name = 'unstructured_policy_pivot_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0.0 的 Series (铁律1: 零值休眠)
        signal = pd.Series(0.0, index=data.index)

        # 检查基础必要字段 (禁止引用 CoreAnchor 数据)
        required_cols = ['usepuindxd', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 1. 恐慌/不确定性指标及衰竭条件 (铁律2: 二阶导数, Anti-Catch-Falling-Knife)
        # 计算 EPU 的 Z-Score (反映宏观政策的极度不确定性/恐慌)
        epu = data['usepuindxd'].ffill()
        epu_mean = epu.rolling(window=252, min_periods=21).mean()
        epu_std = epu.rolling(window=252, min_periods=21).std() + 1e-6
        epu_zscore = (epu - epu_mean) / epu_std
        
        # 衰竭条件: 不确定性处于高位 (>2.0)，且开始边际回落 (小于3日均值)
        epu_exhaustion = (epu_zscore > 2.0) & (epu < epu.rolling(3).mean())

        # 2. 利率预期及曲线的边际突变 (铁律3: 边际变化, 捕捉瞬时脉冲)
        # 2年期美债收益率的5日动量 (降息/加息预期的最敏锐前瞻指标)
        dgs2 = data['dgs2'].ffill()
        dgs2_diff = dgs2.diff(5)
        dgs2_diff_z = (dgs2_diff - dgs2_diff.rolling(252, min_periods=21).mean()) / (dgs2_diff.rolling(252, min_periods=21).std() + 1e-6)
        
        # 收益率曲线5日变化量 (辅助确认形态: 变陡或平坦化)
        t10y2y = data['t10y2y'].ffill()
        t10y2y_diff = t10y2y.diff(5)
        
        # 3. FOMC 情绪得分突变 (阶梯状数据的差分脉冲处理)
        if 'fomc_sentiment' in data.columns:
            fomc = data['fomc_sentiment'].ffill()
            # 严格使用边际变化量(diff)，禁止直接使用绝对水位
            fomc_diff = fomc.diff(5)
            fomc_diff_z = (fomc_diff - fomc_diff.rolling(252, min_periods=21).mean()) / (fomc_diff.rolling(252, min_periods=21).std() + 1e-6)
        else:
            fomc_diff_z = pd.Series(0.0, index=data.index)

        # 4. 非线性特征交叉与脉冲信号生成
        
        # 多头触发 (鸽派突变看多TLT): 
        # 条件A: FOMC极端鸽派突发 (Z > 2.5)
        # 条件B: 2年期极度下行 (Z < -2.5) 且 伴随曲线变陡 (t10y2y_diff > 0, Bull Steepening)
        dovish_shock = (fomc_diff_z > 2.5) | ((dgs2_diff_z < -2.5) & (t10y2y_diff > 0.0))
        
        # 空头触发 (鹰派突变看空TLT):
        # 条件A: FOMC极端鹰派突发 (Z < -2.5)
        # 条件B: 2年期极度上行 (Z > 2.5) 且 伴随曲线平坦/倒挂 (t10y2y_diff < 0, Bear Flattening)
        hawkish_shock = (fomc_diff_z < -2.5) | ((dgs2_diff_z > 2.5) & (t10y2y_diff < 0.0))

        # 信号合成: 政策突变脉冲 + 恐慌极值衰竭确认 (避免主跌浪接飞刀)
        long_signal = dovish_shock & epu_exhaustion
        short_signal = hawkish_shock & epu_exhaustion

        # 赋值 +1.0 / -1.0
        signal.loc[long_signal] = 1.0
        signal.loc[short_signal] = -1.0
        
        # 处理潜在的极端冲突日 (置0)
        conflict = long_signal & short_signal
        signal.loc[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"