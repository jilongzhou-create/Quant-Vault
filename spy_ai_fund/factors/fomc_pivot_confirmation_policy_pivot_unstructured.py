import numpy as np
import pandas as pd

class EpuShockResolutionFactor:
    """EPU Shock and Resolution Pulse (policy_pivot/unstructured)

    逻辑: 采用基于新闻文本的非结构化数据(经济政策不确定性指数EPU)。高频突发的不确定性飙升代表政策风险加剧(冲击, 看空)；而当极度不确定性从高位迅速衰竭并向均值回归时, 代表政策路径明朗, 风险溢价解除(恐慌衰竭, 看多)。
    数据: usepuindxd (Daily News-Based Economic Policy Uncertainty Index for the US)
    输出: +1.0(政策明确，抄底)，-1.0(政策风险骤增，看空)
    触发条件: 1季度(63天)基准下，Z-Score突破+1.5视为突发冲击，从>1.5迅速回落至<0.0视为衰竭解除。信号维持3天，预期Trigger Rate 8% - 12%。
    """

    def __init__(self):
        self.name = 'epu_shock_resolution_pulse_policy_pivot_unstructured'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 如果缺少所需数据，直接返回全 0 序列
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 1. 预处理：周度平滑以消除日内新闻发布噪音，并处理缺失值
        epu = data['usepuindxd'].ffill()
        epu_ma5 = epu.rolling(window=5, min_periods=1).mean()

        # 2. 宏观基准线：1个交易季度 (63天) 作为衡量短期政策预期的锚准
        epu_baseline = epu_ma5.rolling(window=63, min_periods=21).mean()
        epu_std = epu_ma5.rolling(window=63, min_periods=21).std()
        
        # 计算动量变化的Z-Score (符合边际变化铁律)
        epu_z = (epu_ma5 - epu_baseline) / (epu_std + 1e-8)

        # 3. 极值状态记录：过去两周 (10个交易日) 的上下文
        past_10d_max = epu_z.rolling(window=10, min_periods=1).max().shift(1)
        past_10d_min = epu_z.rolling(window=10, min_periods=1).min().shift(1)

        # 4. 看多脉冲：政策不确定性解除 (Resolution)
        # 极值 + 衰竭：近期出现过极端高不确定性(>1.5)，今日彻底跌破基准线(<0.0) -> 恐慌衰竭抄底
        resolution_cross_down = (epu_z < 0.0) & (epu_z.shift(1) >= 0.0)
        buy_trigger = (past_10d_max >= 1.5) & resolution_cross_down

        # 5. 看空脉冲：政策不确定性突发冲击 (Shock)
        # 平静 + 激增：近期环境相对平静(<0.0)，今日突然激增超过极端阈值(>1.5) -> 突发政策风险
        shock_cross_up = (epu_z > 1.5) & (epu_z.shift(1) <= 1.5)
        sell_trigger = (past_10d_min <= 0.0) & shock_cross_up

        # 6. 脉冲展宽：维持3天，确保信号呈现狙击手式的极端短线特征且满足 5%-15% 的 Trigger Rate 铁律
        buy_pulse = buy_trigger.rolling(window=3, min_periods=1).max() > 0
        sell_pulse = sell_trigger.rolling(window=3, min_periods=1).max() > 0

        # 7. 合成信号
        signal = pd.Series(0.0, index=data.index)
        signal[sell_pulse] = -1.0
        signal[buy_pulse] = 1.0  # 优先看多信号(符合 SPY 长牛下防接飞刀后的确定性买点)

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"