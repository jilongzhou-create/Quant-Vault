import numpy as np
import pandas as pd

class FomcEpuPivotPulseFactor:
    """FomcEpuPivotPulse (policy_pivot/unstructured)

    逻辑: 捕捉美联储货币政策预期的剧变以及宏观经济政策不确定性(EPU)的靴子落地效应。当FOMC情绪得分发生跳跃，或政策不确定性指数在历史高位衰竭时，触发短线交易脉冲。
    数据: fomc_sentiment, usepuindxd
    输出: 鸽派突变或EPU恐慌落地看多(+1.0)，鹰派突变看空(-1.0)
    触发条件: FOMC情绪单日变化>0.15或EPU Z-Score>2.0且连跌3天，信号持续3天，预期Trigger Rate约在5%-12%
    """

    def __init__(self, fomc_diff_threshold=0.15, epu_z_threshold=2.0, signal_window=3):
        self.name = 'fomc_epu_pivot_pulse'
        self.fomc_diff_threshold = fomc_diff_threshold
        self.epu_z_threshold = epu_z_threshold
        self.signal_window = signal_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        has_fomc = 'fomc_sentiment' in data.columns
        has_epu = 'usepuindxd' in data.columns
        
        if not has_fomc and not has_epu:
            return signal

        bull_signal = pd.Series(False, index=data.index)
        bear_signal = pd.Series(False, index=data.index)

        if has_fomc:
            fomc = data['fomc_sentiment'].ffill()
            # 【边际变化铁律】: 严格对待低频阶梯状数据，只能在改变发生的瞬间捕捉跃迁动量
            fomc_diff = fomc.diff()
            
            # 鸽派突变: 声明态度剧烈向鸽派转向
            fomc_bull = fomc_diff >= self.fomc_diff_threshold
            # 鹰派突变: 声明态度剧烈向鹰派转向
            fomc_bear = fomc_diff <= -self.fomc_diff_threshold
            
            # 【零值休眠铁律】: 仅在突变日及随后极短的窗口期内输出信号 (脉冲式)
            fomc_bull_ext = fomc_bull.rolling(window=self.signal_window, min_periods=1).max() > 0
            fomc_bear_ext = fomc_bear.rolling(window=self.signal_window, min_periods=1).max() > 0
            
            bull_signal = bull_signal | fomc_bull_ext
            bear_signal = bear_signal | fomc_bear_ext

        if has_epu:
            epu = data['usepuindxd'].ffill()
            # 计算政策不确定性指数的年度Z-score
            epu_mean = epu.rolling(window=252, min_periods=60).mean()
            epu_std = epu.rolling(window=252, min_periods=60).std()
            epu_z = (epu - epu_mean) / (epu_std + 1e-8)
            
            # 【二阶导数铁律】: 不确定性高企时直接买入会接飞刀，必须满足“极值 + 衰竭”
            epu_extreme = epu_z >= self.epu_z_threshold
            epu_falling = epu.diff(3) < 0
            
            # 政策恐慌情绪已经见顶且开始退散，对于长牛美股是非常确定的抄底买点
            epu_bull = epu_extreme & epu_falling
            bull_signal = bull_signal | epu_bull
            
        signal[bull_signal] = 1.0
        signal[bear_signal] = -1.0
        
        # 多空冲突时保持中立，防止无意义损耗
        signal[bull_signal & bear_signal] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(fomc_diff_threshold={self.fomc_diff_threshold}, epu_z_threshold={self.epu_z_threshold}, signal_window={self.signal_window})"