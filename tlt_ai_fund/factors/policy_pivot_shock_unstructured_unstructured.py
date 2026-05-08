import numpy as np
import pandas as pd

class PolicyPivotShockFactor:
    """政策预期突变 (Policy Pivot Shock)

    逻辑: 捕捉短端利率(对政策最敏感的 dgs2)边际骤变引发的联储政策重定价预期。当2年期美债收益率暴跌且曲线变陡(Bull Steepening)时，意味着降息预期突增，利多美债(TLT)。为避免在最剧烈波动期逆势接飞刀，必须等短端利率的下杀/逼空动能开始衰竭时，才发出一次性脉冲信号。
    数据: dgs2 (2年期美债收益率), t10y2y (10年-2年美债利差)
    触发: dgs2 的 5日变化量 Z-Score 极端(绝对值>1.8) + t10y2y 形态验证 + dgs2 动能衰竭(二阶导反转)
    输出: +1.0 狙击降息预期突增的右侧多头, -1.0 狙击加息预期突增的右侧空头。其余常态日严格休眠为 0.0。
    """

    def __init__(self, window=5, zscore_window=252, z_threshold=1.8):
        self.name = 'policy_pivot_shock_pulse'
        self.window = window
        self.zscore_window = zscore_window
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 安全性: 检查必要字段是否存在
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return pd.Series(0.0, index=data.index)

        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝对禁止使用收益率水位的绝对值，必须使用滚动变化量
        dgs2_diff = dgs2.diff(self.window)
        t10y2y_diff = t10y2y.diff(self.window)

        # 动态 Z-Score 计算，衡量边际变化的极端程度
        roll_mean = dgs2_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).mean()
        roll_std = dgs2_diff.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).std()
        roll_std = roll_std.replace(0, np.nan)  # 防止除以 0
        
        zscore = (dgs2_diff - roll_mean) / roll_std

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 必须等待重定价的动能出现边际衰竭，才允许介入
        dgs2_diff_shifted = dgs2_diff.shift(1)
        
        # 多头衰竭: 收益率暴跌的幅度开始收窄 (负值向上反弹)
        exhaustion_long = dgs2_diff > dgs2_diff_shifted
        
        # 空头衰竭: 收益率暴涨的幅度开始收窄 (正值向下回落)
        exhaustion_short = dgs2_diff < dgs2_diff_shifted

        # 极值条件 + 结构验证 + 衰竭确认
        # 看多TLT (降息突变): 短端暴跌极值 + 曲线牛陡(Bull Steepening) + 跌势衰竭
        long_cond = (zscore < -self.z_threshold) & (t10y2y_diff > 0) & exhaustion_long

        # 看空TLT (加息突变): 短端暴涨极值 + 曲线熊平(Bear Flattening) + 涨势衰竭
        short_cond = (zscore > self.z_threshold) & (t10y2y_diff < 0) & exhaustion_short

        # 铁律1: 零值休眠 (Sniper Pulse)
        signal = pd.Series(0.0, index=data.index)
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold})"