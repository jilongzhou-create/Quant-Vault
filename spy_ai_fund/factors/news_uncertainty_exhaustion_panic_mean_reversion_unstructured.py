import numpy as np
import pandas as pd

class NewsUncertaintyExhaustionFactor:
    """EPU 新闻恐慌见顶回归 (panic_mean_reversion/unstructured)

    逻辑: 使用基于非结构化新闻文本提取的 EPU(经济政策不确定性) 日度指数。在长牛的美股市场中，由于突发新闻事件驱动的极端恐慌往往是反向做多的绝佳买点；而轻微发酵的恐慌则是引发趋势恶化的“钝刀子”。当 EPU 异常飙升超过1.5倍标准差，并首次出现日度边际回落时，标志着突发新闻发酵完毕/靴子落地(恐慌衰竭)，触发极值做多脉冲；当 EPU 刚脱离常态并上穿0.8倍标准差时，提示市场情绪恶化，触发脉冲看空。
    数据: usepuindxd (Daily Economic Policy Uncertainty Index)
    输出: +1.0 表示极端恐慌衰竭的做多买点, -1.0 表示轻微恐慌开始蔓延的看空起点
    触发条件: 狙击型脉冲。做多 = 252日Z-Score > 1.5 且 3日均线出现首日倒V反转下降；做空 = Z-Score 单日向上穿越 0.8。预期 Trigger Rate 控制在 5%-10% 之间。
    """

    def __init__(self):
        self.name = 'news_uncertainty_exhaustion'
        self.smooth_window = 3      # 消除单日新闻报道噪音的短期平滑窗口
        self.zscore_window = 252    # 年度宏观常态基准线(一年的交易日)
        self.extreme_panic_threshold = 1.5  # 显著的黑天鹅或极端不确定性高点阈值 (约前6.7%)
        self.creeping_panic_threshold = 0.8 # 恐慌开始发酵的趋势脱离预警位 (约前21%)

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值休眠信号
        signal = pd.Series(0.0, index=data.index, name=self.name)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 1. 填充可能缺失的新闻不确定性数据 (前向填充)
        epu = data['usepuindxd'].ffill()
        
        # 2. 短期平滑以过滤高频的单日新闻脉冲噪音，寻找周内连续发酵特征
        smoothed_epu = epu.rolling(window=self.smooth_window).mean()
        
        # 3. 计算滚动的 Z-Score, 衡量当下的宏观不确定性脱离常态的程度
        roll_mean = smoothed_epu.rolling(window=self.zscore_window).mean()
        roll_std = smoothed_epu.rolling(window=self.zscore_window).std()
        
        # 避免除以 0 导致错误
        zscore = (smoothed_epu - roll_mean) / roll_std.replace(0, np.nan)
        zscore = zscore.fillna(0.0)
        
        # 4. 二阶导动量 (今日变化率边际变化)
        epu_diff = smoothed_epu.diff(1)
        
        # 5. 买入信号逻辑 (恐慌极值 + 衰竭确认)
        # 昨日已处于极端高位阈值以上且依然在发酵恶化，今日恐慌突然减弱(diff首日由正转负)，形成明确的倒V衰竭转折
        bull_cond = (
            (zscore.shift(1) > self.extreme_panic_threshold) & 
            (epu_diff < 0) & 
            (epu_diff.shift(1) > 0)
        )
        
        # 6. 卖出信号逻辑 (轻微恐慌蔓延确认)
        # 不确定性刚刚向上跨越警戒阈值，钝刀割肉杀估值的开端（边际恶化的脉冲瞬间）
        bear_cond = (
            (zscore.shift(1) <= self.creeping_panic_threshold) & 
            (zscore > self.creeping_panic_threshold)
        )
        
        # 7. 分配脉冲信号 (每日输出限定在极短瞬间的 0 或 +/- 1)
        signal.loc[bull_cond] = 1.0
        signal.loc[bear_cond] = -1.0
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(smooth_win={self.smooth_window}, zscore_win={self.zscore_window}, ext_thr={self.extreme_panic_threshold}, crp_thr={self.creeping_panic_threshold})"