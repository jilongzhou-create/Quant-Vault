import numpy as np
import pandas as pd

class PolicyPivotBullSteepeningFactor:
    """政策预期突变之牛市变陡脉冲因子 (Policy Pivot Shock)

    逻辑: 捕捉美联储政策预期由紧转松的瞬间。短端利率(dgs2)比长端对政策变化更敏感，当dgs2短时暴跌且伴随收益率曲线(t10y2y)急速变陡时，标志着宽松预期以突发冲击的形式被Price-in(典型的Bull Steepening)。只有当剧烈的冲击动能出现衰竭时才触发交易，避免在主跌浪接飞刀。
    数据: dgs2 (2年期美债收益率), t10y2y (10-2年利差)
    触发: dgs2 5日动量极度下行 (Z-Score < -2.5) + t10y2y 5日急速拉升变陡 (Z-Score > 2.0) + dgs2下行动能衰竭 (diff > shift)
    输出: +1.0表示宽松突变(看多美债)的狙击手脉冲, -1.0表示紧缩突变(看空美债), 否则常态为0.0
    """

    def __init__(self, window=5, z_window=252, z_threshold=2.5, steep_threshold=2.0):
        self.name = 'unstructured_policy_pivot_bull_steepening_shock'
        self.window = window
        self.z_window = z_window
        self.z_threshold = z_threshold
        self.steep_threshold = steep_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查所需数据是否齐全
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝对不使用绝对水位, 也不管收益率曲线是否倒挂, 只关注动量的边际突变
        dgs2_diff = dgs2.diff(self.window)
        t10y2y_diff = t10y2y.diff(self.window)
        
        # 计算滚动的 Z-Score
        dgs2_mean = dgs2_diff.rolling(window=self.z_window, min_periods=self.z_window // 2).mean()
        dgs2_std = dgs2_diff.rolling(window=self.z_window, min_periods=self.z_window // 2).std()
        dgs2_z = (dgs2_diff - dgs2_mean) / (dgs2_std + 1e-8)
        
        t10y2y_mean = t10y2y_diff.rolling(window=self.z_window, min_periods=self.z_window // 2).mean()
        t10y2y_std = t10y2y_diff.rolling(window=self.z_window, min_periods=self.z_window // 2).std()
        t10y2y_z = (t10y2y_diff - t10y2y_mean) / (t10y2y_std + 1e-8)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 动能衰竭: 今天的动量变化开始反转（下行势头开始收缩或上行势头开始触顶）
        dgs2_diff_shifted = dgs2_diff.shift(1)
        bull_steepening_exhaustion = (dgs2_diff > dgs2_diff_shifted)  # dgs2暴跌的负值缩小 = 动能衰竭
        bear_flattening_exhaustion = (dgs2_diff < dgs2_diff_shifted)  # dgs2暴涨的正值缩小 = 动能衰竭
        
        # 多头脉冲: dgs2动能极度下行 (预期降息) + t10y2y急速拉升 (Bull Steepening) + 下行衰竭确认
        long_cond = (dgs2_z < -self.z_threshold) & (t10y2y_z > self.steep_threshold) & bull_steepening_exhaustion
        
        # 空头脉冲: dgs2动能极度上行 (超预期紧缩) + t10y2y急速下坠 (Bear Flattening) + 上行衰竭确认
        short_cond = (dgs2_z > self.z_threshold) & (t10y2y_z < -self.steep_threshold) & bear_flattening_exhaustion
        
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 初始赋值为 0.0，仅在极端事件发生 + 衰竭满足时给与信号
        signal = pd.Series(0.0, index=data.index)
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_window={self.z_window}, z_threshold={self.z_threshold})"