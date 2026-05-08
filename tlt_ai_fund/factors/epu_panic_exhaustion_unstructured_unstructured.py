import numpy as np
import pandas as pd

class FomcYieldConfirmationFactor:
    """FOMC Policy Yield Confirmation (unstructured/unstructured)

    逻辑: 捕捉美联储政策预期突变，并要求收益率发生同向衰竭确认。只有当 FOMC 情绪得分出现 2.5σ 以上的极端鸽派跳跃，并且对政策最敏感的 2年期美债(dgs2) 顺势跌破 5日均线时，才确认降息预期被市场真正 Pricing，输出做多脉冲；鹰派突变同理反向。通过收益率确认，解决“买预期卖事实”导致的 CondIC 为负的问题。
    数据: fomc_sentiment (FOMC情绪得分), dgs2 (2年期美债)
    触发: 极值 (fomc_diff5 Z-Score > 2.5) + 衰竭/确认 (dgs2 < 5日均线)
    输出: 狙击手脉冲信号 [-1.0, 1.0]，非触发日严格为 0.0，维持10天以满足 5%-15% 触发率目标
    """

    def __init__(self):
        self.name = 'fomc_yield_confirmation_unstructured_unstructured'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值休眠信号
        signal = pd.Series(0.0, index=data.index)

        # 缺失字段保护
        if 'fomc_sentiment' not in data.columns or 'dgs2' not in data.columns:
            return signal

        # 1. 数据预处理
        fomc = data['fomc_sentiment'].ffill().fillna(0)
        dgs2 = data['dgs2'].ffill()

        # 铁律3: 边际变化 (Marginal Change Only)
        # 禁止直接用绝对水位，使用 5日变化量捕捉预期跳跃
        fomc_diff5 = fomc.diff(5).fillna(0)

        # 2. 极值状态计算 (Z-Score > 2.5)
        # 使用 504日(约2年)滚动窗口，设定标准差下限防0除(因为多数时间Diff为0)
        mean_504 = fomc_diff5.rolling(window=504, min_periods=60).mean().fillna(0)
        std_504 = fomc_diff5.rolling(window=504, min_periods=60).std().fillna(0.15)
        std_504 = np.maximum(std_504, 0.15)  # 设定底层阈值，确保微小跳跃不会触发
        fomc_z = (fomc_diff5 - mean_504) / std_504

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 短期美债收益率必须出现同向衰竭与趋势确认，严禁单边接飞刀(应对"Sell the fact")
        dgs2_ma5 = dgs2.rolling(window=5, min_periods=1).mean()
        dgs2_falling = dgs2 < dgs2_ma5  # 收益率下行 = 美债价格反弹确认
        dgs2_rising = dgs2 > dgs2_ma5   # 收益率上行 = 美债价格下跌确认

        # 铁律1: 零值休眠 (Sniper Pulse)
        # 极端事件 + 衰竭确认 双满足时才打出非零脉冲
        buy_pulse = (fomc_z > 2.5) & dgs2_falling
        sell_pulse = (fomc_z < -2.5) & dgs2_rising

        # 将脉冲维持 10 天，确保有效捕捉行情波段并使 Trigger Rate 落在 5%-15%
        buy_hold = buy_pulse.rolling(window=10, min_periods=1).max() > 0
        sell_hold = sell_pulse.rolling(window=10, min_periods=1).max() > 0

        # 生成信号
        signal[buy_hold] = 1.0
        signal[sell_hold] = -1.0
        
        # 冲突保护
        signal[buy_hold & sell_hold] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"