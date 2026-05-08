import numpy as np
import pandas as pd

class EpuPanicMeanReversionFactor:
    """政策不确定性恐慌均值回归因子 (panic_mean_reversion/unstructured)

    逻辑: 基于美国每日经济政策不确定性新闻指数(usepuindxd，非结构化NLP数据)挖掘SPY的恐慌与均值回归转折点。根据SPY长牛且均值回归的物理特性：当政策恐慌在季度维度达到极端水平(Z-Score>1.5)且动量边际衰竭时，是极佳的“恐慌耗竭”抄底买点；反之，若在非极端区域内出现短期(3日)恐慌跳跃发酵，属于“钝刀割肉”式的波段看空信号。
    数据: [usepuindxd]
    输出: 极端恐慌见顶衰竭输出强看多(+1.0)，非极端恐慌情绪快速积累突变看空(-1.0)
    触发条件: 买入脉冲要求前日到达极端看空区且今日动量反转(防接飞刀)；卖出脉冲要求3日增幅跳跃过界且处于酝酿期。预期 Trigger Rate 控制在 8%-12%。
    """

    def __init__(self):
        self.name = 'epu_panic_mean_reversion_unstructured'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 数据缺失与空值保护
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        epu = data['usepuindxd'].ffill()
        if epu.isna().all():
            return pd.Series(0.0, index=data.index, name=self.name)

        signal = pd.Series(0.0, index=data.index)

        # 核心物理参数 - 具有经济学含义的阈值
        window = 63  # 63个交易日(单季度)，适应中期宏观预期与政策周期的稳态
        
        # 计算EPU的季度滚动统计量 (均值回归特征)
        epu_mean = epu.rolling(window=window, min_periods=window//2).mean()
        epu_std = epu.rolling(window=window, min_periods=window//2).std().replace(0, 1e-5)
        epu_z = (epu - epu_mean) / epu_std

        # 计算日频差分和3日动量 (边际变化铁律: 不直接使用绝对值)
        epu_diff = epu.diff()
        epu_3d_diff = epu.diff(3)

        # ==========================================
        # 1. 狙击脉冲 - 强力抄底看多(+1.0): 极值 + 衰竭
        # ==========================================
        # 极度恐慌条件: 前一日不确定性达到了罕见的恐慌极值(>1.5 StdDev)
        is_extreme = epu_z > 1.5
        # 二阶导数衰竭拐点: 昨天政策风险还在发酵(diff>0)，但今天情绪已实质性回落(diff<0) -> 恐慌顶峰过去
        panic_exhaustion = is_extreme.shift(1) & (epu_diff < 0) & (epu_diff.shift(1) > 0)

        # ==========================================
        # 2. 狙击脉冲 - 趋势恶化看空(-1.0): 恐慌突发积累
        # ==========================================
        # 轻度恐慌发酵: 不确定性指数在3天内跳升超过1.2倍季度标准差
        mild_surge = epu_3d_diff > (1.2 * epu_std)
        # 脉冲锁定铁律: 只在越过动量阈值的当天触发一次，且要求环境尚未进入极度恐慌极值(Z<1.5, 防止过早抄底接飞刀)
        mild_panic_pulse = mild_surge & (~mild_surge.shift(1).fillna(False)) & (epu_z < 1.5)

        # 信号合成 (同一天同时发生优先视为衰竭反弹信号，符合SPY向上趋势属性)
        signal[mild_panic_pulse] = -1.0
        signal[panic_exhaustion] = 1.0 

        signal.name = self.name
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}()"