import numpy as np
import pandas as pd

class FearGreedVixExhaustionFactor:
    """Fear Greed VIX Exhaustion Pulse (panic_mean_reversion/nonlinear)

    逻辑: 结合CNN恐慌贪婪指数(Fear & Greed)与VIX。美股具有长牛和均值回归特性，当情绪极度悲观且VIX处于高位时，若二者同时出现边际改善（VIX回落，恐慌指数回升），即刻触发强烈看多的抄底脉冲；若VIX温和上升且情绪恶化，则判定为轻度恐慌的'钝刀割肉'状态，触发看空脉冲。
    数据: [vixcls, fear_greed]
    输出: [+1.0 看多(恐慌极值+衰竭), -1.0 看空(温和恐慌恶化), 0.0 常态休眠]
    触发条件: [VIX Z-Score > 1.2且恐慌指数<35且日内双双改善，触发+1.0；VIX Z-Score介于0.0~1.2且日内恶化，触发-1.0，预期Trigger Rate 8-12%]
    """

    def __init__(self):
        self.name = 'fear_greed_vix_exhaustion_pulse_panic_mean_reversion_nonlinear'
        self.vix_window = 252

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 零值休眠铁律: 默认返回 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 数据缺失保护
        if 'vixcls' not in data.columns or 'fear_greed' not in data.columns:
            return signal

        vix = data['vixcls'].ffill()
        fg = data['fear_greed'].ffill()

        valid_mask = vix.notna() & fg.notna()

        # 计算 VIX 的 252 日 Z-Score
        vix_mean = vix.rolling(window=self.vix_window, min_periods=60).mean()
        vix_std = vix.rolling(window=self.vix_window, min_periods=60).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-6)

        # 二阶导数铁律：获取每日边际变化
        vix_diff = vix.diff()
        fg_diff = fg.diff()

        # 看多脉冲 (+1.0)：极端恐慌极值 + 边际衰竭 (抄底买入)
        # 1. VIX 处于相对高位 (Z > 1.2)
        # 2. 贪婪恐慌指数处于恐慌区间 (Fear & Greed < 35)
        # 3. 衰竭确认: VIX当天开始回落 (vix_diff < 0) 且情绪指数当天开始回升 (fg_diff > 0)
        long_cond = (vix_z > 1.2) & (fg < 35) & (vix_diff < 0) & (fg_diff > 0)

        # 看空脉冲 (-1.0)：轻微恐慌 + 趋势恶化 (钝刀割肉)
        # 1. VIX 在均值之上但未到极值区间 (0.0 < Z <= 1.2)
        # 2. 趋势恶化: VIX 当天上升 (vix_diff > 0) 且情绪当天下降 (fg_diff < 0)
        short_cond = (vix_z > 0.0) & (vix_z <= 1.2) & (vix_diff > 0) & (fg_diff < 0)

        signal[long_cond & valid_mask] = 1.0
        signal[short_cond & valid_mask] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"