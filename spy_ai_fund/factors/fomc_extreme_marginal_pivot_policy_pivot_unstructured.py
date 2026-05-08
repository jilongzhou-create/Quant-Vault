import numpy as np
import pandas as pd

class FomcExtremeMarginalPivotFactor:
    """Fomc Extreme Marginal Pivot Factor (policy_pivot/unstructured)

    逻辑: 捕捉美联储在明确的鹰派或鸽派周期中，突然出现超预期边际逆转的瞬间（最坏时期过去，或宽松红利期结束）。当会前处于鹰派且出现强烈的边际鸽派变化时看多美股，反之看空。信号仅在预期剧变的极短窗口内发酵。
    数据: [fomc_sentiment]
    输出: 1.0 看多（边际转鸽抢跑），-1.0 看空（边际转鹰杀估值），其余时间 0.0
    触发条件: 会前情绪 < 0 且边际改善(diff) >= 0.2 触发看多脉冲；会前情绪 > 0 且边际恶化(diff) <= -0.2 触发看空脉冲。每次脉冲维持 5 个交易日，预期 Trigger Rate 控制在 8% - 12%。
    """

    def __init__(self, diff_threshold=0.2, hold_days=5):
        self.name = 'fomc_extreme_marginal_pivot'
        self.diff_threshold = diff_threshold
        self.hold_days = hold_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 缺失字段直接返回全 0 休眠信号
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # FOMC数据为阶梯状低频前向填充数据，必须提取边缘变化
        fomc = data['fomc_sentiment'].ffill()
        fomc_diff = fomc.diff().fillna(0.0)
        fomc_prev = fomc.shift(1).fillna(0.0)

        # 狙击手条件1：鹰派深水区的边际转暖 (看多)
        buy_trigger = (fomc_prev < 0.0) & (fomc_diff >= self.diff_threshold)
        
        # 狙击手条件2：鸽派高水位的边际转冷 (看空)
        sell_trigger = (fomc_prev > 0.0) & (fomc_diff <= -self.diff_threshold)

        # 转换为数值型单点脉冲
        buy_pulse = buy_trigger.astype(float)
        sell_pulse = sell_trigger.astype(float)

        # 信号在极短窗口内（hold_days）存续发酵，随后休眠归零
        buy_signal = buy_pulse.rolling(window=self.hold_days, min_periods=1).max()
        sell_signal = sell_pulse.rolling(window=self.hold_days, min_periods=1).max()

        # 组合脉冲信号，限制在 [-1.0, 1.0]
        signal = buy_signal - sell_signal
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(diff_threshold={self.diff_threshold}, hold_days={self.hold_days})"