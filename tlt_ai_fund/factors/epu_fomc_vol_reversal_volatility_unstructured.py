import numpy as np
import pandas as pd

class EpuFomcVolReversalFactor:
    """FomcEpuVolReversalFactor (volatility/unstructured)

    逻辑: 结合基于新闻文本的政策不确定性(EPU)极值衰竭与央行文本(FOMC)的情绪动量。当宏观不确定性狂飙后见顶回落时，市场迎来重新定价。若此时央行处于边际转鸽周期(FOMC diff > 0)，不确定性消除将推升美债(避险资金追逐安全Carry)；若处于转鹰周期，确定性加息"靴子落地"会导致美债遭遇抛售脉冲。
    数据: usepuindxd (日频新闻经济政策不确定性), fomc_sentiment (日频FOMC文本鹰鸽得分)。
    触发: usepuindxd 63日Z-Score > 1.25 且 开始衰竭(低于3日均值) 且 FOMC季度边际变化绝对值 > 0.05。
    输出: 脉冲型，常态为0.0，极端组合下输出+1.0/-1.0。
    """

    def __init__(self):
        self.name = 'epu_fomc_vol_reversal'
        self.lookback_window = 63  # 季度窗口，捕捉局部博弈并确保覆盖至少一次FOMC决议
        self.zscore_threshold = 1.25 # 单尾异常阈值(约前10%极值)，保证Target Trigger Rate在合理脉冲区间
        self.fomc_momentum_threshold = 0.05 # 过滤微小噪音，要求实质性的鹰鸽边际转向
        self.exhaustion_window = 3 # 短期反转确认窗口

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (Sniper Pulse)
        signal = pd.Series(0.0, index=data.index)

        if 'usepuindxd' not in data.columns or 'fomc_sentiment' not in data.columns:
            return signal

        epu = data['usepuindxd'].ffill()
        fomc = data['fomc_sentiment'].ffill()

        # 计算 EPU 的 Z-Score (季度滚动，适应不同宏观周期的基准水位的抬升)
        epu_mean = epu.rolling(window=self.lookback_window, min_periods=21).mean()
        epu_std = epu.rolling(window=self.lookback_window, min_periods=21).std()
        
        epu_std = epu_std.replace(0, np.nan)
        epu_zscore = (epu - epu_mean) / epu_std

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 条件1: 不确定性处于极端狂飙状态
        is_high_vol = epu_zscore > self.zscore_threshold
        # 条件2: 不确定性开始衰竭回落 (不再创新高，跌破近期均值)
        is_exhausting = epu < epu.rolling(window=self.exhaustion_window).mean()

        # 铁律3: 边际变化 (Marginal Change Only)
        # 禁止使用绝对值，使用季度维度的 .diff() 捕捉预期的边际改变
        fomc_momentum = fomc.diff(self.lookback_window)

        # 组合脉冲信号
        # 转鸽基调下，宏观恐慌衰竭 -> 央行庇护预期接管，利好美债
        bull_condition = is_high_vol & is_exhausting & (fomc_momentum > self.fomc_momentum_threshold)
        
        # 转鹰基调下，宏观恐慌衰竭 -> 紧缩靴子落地，避险消退，利空美债
        bear_condition = is_high_vol & is_exhausting & (fomc_momentum < -self.fomc_momentum_threshold)

        signal[bull_condition] = 1.0
        signal[bear_condition] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback={self.lookback_window}, z={self.zscore_threshold})"