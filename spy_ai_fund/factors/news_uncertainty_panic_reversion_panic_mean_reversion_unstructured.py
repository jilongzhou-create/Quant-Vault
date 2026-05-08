import numpy as np
import pandas as pd

class NewsUncertaintyPanicReversionFactor:
    """新闻政策不确定性恐慌反转因子 (panic_mean_reversion/unstructured)

    逻辑: 跟踪每日新闻文本蕴含的经济政策不确定性(EPU)指数。极端高企的新闻恐慌(Z-Score > 2.5)在见顶回落瞬间触发看多(反飞刀均值回归)；而处于上升通道的温和恐慌(Z-Score > 1.0且破新高)表明不确定性正在发酵恶化，触发看空脉冲。
    数据: usepuindxd (Daily News implied Economic Policy Uncertainty Index)
    输出: +1.0 强烈看多(极端恐慌衰竭)，-1.0 看空(不确定性发酵恶化)，0.0 休眠
    触发条件: 买入: 63日Z-Score > 2.5 且 今日EPU下降且低于3日均值；卖出: 63日Z-Score 在 1.0~2.0之间且首次突破5日新高，预期 Trigger Rate 5%-12%。
    """

    def __init__(self, window=63, z_extreme=2.5, z_mild=1.0):
        self.name = 'news_uncertainty_panic_reversion_unstructured'
        self.window = window
        self.z_extreme = z_extreme
        self.z_mild = z_mild

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        epu = data['usepuindxd'].ffill()
        
        # 季度级别均值回归基准
        roll_mean = epu.rolling(window=self.window).mean()
        roll_std = epu.rolling(window=self.window).std()
        z_score = (epu - roll_mean) / (roll_std + 1e-8)

        epu_diff = epu.diff()
        epu_ma3 = epu.rolling(window=3).mean()
        epu_max5 = epu.shift(1).rolling(window=5).max()

        # 核心二阶导数铁律：极值 + 衰竭
        # 政策不确定性处于局部极端高位，且今日停止飙升（diff < 0），并且跌破短期均线
        condition_extreme_panic = z_score > self.z_extreme
        condition_exhaustion = (epu_diff < 0) & (epu < epu_ma3)
        buy_pulse = condition_extreme_panic & condition_exhaustion

        # 边际恶化逻辑：恐慌正在缓慢发酵 (钝刀割肉)
        # 尚未达到极端值，但在高位警戒区间，且新闻恐慌度突破短期新高
        condition_mild_panic = (z_score > self.z_mild) & (z_score <= 2.0)
        condition_worsening = epu > epu_max5
        # 过滤连续脉冲，要求前一天不是突破状态（即仅在脉冲瞬间触发）
        epu_max5_prev = epu.shift(2).rolling(window=5).max()
        condition_just_breakout = epu.shift(1) <= epu_max5_prev
        sell_pulse = condition_mild_panic & condition_worsening & condition_just_breakout

        signal = pd.Series(0.0, index=data.index)
        signal.loc[buy_pulse] = 1.0
        signal.loc[sell_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_extreme={self.z_extreme}, z_mild={self.z_mild})"