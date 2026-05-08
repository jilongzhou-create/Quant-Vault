import numpy as np
import pandas as pd

class FomcPivotShockPulseFactor:
    """政策预期突变脉冲因子 (unstructured/unstructured)

    逻辑: FOMC 声明措辞的突变往往预示美联储货币政策框架的系统性转向 (Pivot)。
          本因子旨在捕捉 fomc_sentiment 的极端阶梯跳跃。由于情绪得分是阶梯状低频跳跃数据，
          稳态下输出必须保持 0 休眠。仅当捕捉到异于长年常规变动的高级别跳跃，并在跳跃动能衰竭时输出短暂波段脉冲。
    数据: fomc_sentiment
    触发: 单日边际变动量(diff)的 252 日 Z-Score 绝对值 > 2.5，且次日该变动发生动能衰竭(趋于平息)。
    输出: 脉冲信号，+1.0 (鸽派转向利多美债), -1.0 (鹰派转向利空美债)
    """

    def __init__(self):
        self.name = 'fomc_pivot_shock_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号必须为 pd.Series(0.0) 确保常态休眠
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            signal.name = self.name
            return signal
            
        # 严格铁律3: 边际变化 (Marginal Change Only)
        # 严禁使用绝对水位！只使用一阶边际差分来捕获瞬间冲击
        fomc_series = data['fomc_sentiment'].ffill()
        delta = fomc_series.diff().fillna(0.0)
        
        # 计算长期政策变动的波动率基准 (252个交易日，即过去一年的会议常态波动)
        roll_mean = delta.rolling(window=252, min_periods=21).mean()
        roll_std = delta.rolling(window=252, min_periods=21).std()
        
        # 提取极值 Z-Score，使用 1e-6 避免纯 0 期间除以零错误
        zscore = (delta - roll_mean) / (roll_std + 1e-6)
        
        # 捕捉极端跳跃瞬间: 超出年度变动率分布的 2.5 个标准差
        is_dove_extreme = (zscore > 2.5) & (delta > 0)
        is_hawk_extreme = (zscore < -2.5) & (delta < 0)
        
        # 严格铁律2: 二阶导数反飞刀 (Anti-Catch-Falling-Knife)
        # 对于阶梯数据，必须等突变的脉冲峰值在次日自然回落（增量缩小或为0）才视为新的稳态形成
        # 即条件为：昨日是极端跳跃，且今日的变化幅度收敛（动量衰竭）
        dove_exhaustion_confirmed = is_dove_extreme.shift(1).fillna(False) & (delta.abs() < delta.shift(1).abs())
        hawk_exhaustion_confirmed = is_hawk_extreme.shift(1).fillna(False) & (delta.abs() < delta.shift(1).abs())
        
        # 严格铁律1: 零值休眠 (Sniper Pulse)
        # 事件进入衰竭确认后，持续极短几天（5天 / 一周）输出信号，给予市场 Price-in 缓冲期
        # 目标触发率确保被控制在合理较低频但不会被完全埋没的 5%-15% 区间
        pulse_duration = 5
        dove_pulse = dove_exhaustion_confirmed.rolling(window=pulse_duration, min_periods=1).max() > 0
        hawk_pulse = hawk_exhaustion_confirmed.rolling(window=pulse_duration, min_periods=1).max() > 0
        
        # 非连续性脉冲输出赋值
        signal[dove_pulse] = 1.0
        signal[hawk_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"