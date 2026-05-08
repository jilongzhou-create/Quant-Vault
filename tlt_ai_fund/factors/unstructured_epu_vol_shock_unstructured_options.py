import numpy as np
import pandas as pd

class UnstructuredEpuVolShockFactor:
    """Unstructured EPU Vol Shock (unstructured/options)

    逻辑: 将日频经济政策不确定性指数(usepuindxd)视为非结构化文本领域的“隐含波动率(IV)”。
    通过计算EPU的边际突变(5日变化量)捕捉宏观避险情绪的爆发。当政策恐慌向上极度飙升(Z-Score > 2.5)且
    动能随后死叉下穿均线时，标志着恐慌见顶被市场Price-in，避险资金准备流出美债，输出 -1.0 脉冲看空；
    当极度自满(Z-Score < -2.5)且动能金叉反转升温时，代表政策黑天鹅风险重燃，避险需求骤升，输出 +1.0 脉冲看多。
    严格遵守边际变化、类期权波动率极值与动能衰竭三大铁律。
    数据: usepuindxd (经济政策不确定性指数, 纯文本挖掘的Unstructured代理变量)
    触发: 近3日Z-Score最大值 > 2.5 且今日动能死叉3日均线 -> -1.0；近3日Z-Score最小值 < -2.5 且动能金叉均线 -> +1.0
    输出: 极低频脉冲型信号 [-1.0, 1.0]
    """

    def __init__(self, diff_window=5, z_window=252, smooth_window=3):
        self.name = 'unstructured_epu_vol_shock'
        self.diff_window = diff_window
        self.z_window = z_window
        self.smooth_window = smooth_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (默认全0的狙击手脉冲)
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 1. 提取EPU指数并前向填充处理缺失值
        epu = data['usepuindxd'].ffill()
        
        # 2. 边际变化铁律: 计算预期改变瞬间的动量变化 (禁止使用低频/连续水平绝对值)
        epu_diff = epu.diff(self.diff_window)
        
        # 3. 计算Z-Score以衡量突变极值 (类期权定价中的Volatility Shock)
        roll_mean = epu_diff.rolling(window=self.z_window, min_periods=self.z_window // 2).mean()
        roll_std = epu_diff.rolling(window=self.z_window, min_periods=self.z_window // 2).std()
        
        # 避免除零异常
        roll_std = roll_std.replace(0.0, np.nan)
        z_score = (epu_diff - roll_mean) / roll_std
        
        # 4. 二阶导数衰竭铁律 (Anti-Catch-Falling-Knife)
        # 计算动能的平滑均线，捕捉反转点
        epu_diff_smooth = epu_diff.rolling(window=self.smooth_window).mean()
        
        # 识别近3日内是否发生过极值 (容忍极值点与死叉/金叉点错位1-2天)
        recent_extreme_high = z_score.rolling(window=self.smooth_window).max() > 2.5
        recent_extreme_low = z_score.rolling(window=self.smooth_window).min() < -2.5
        
        # 严格捕捉动能反转的确切瞬间(交叉日), 确保只在拐点当天生成单一脉冲
        cross_down = (epu_diff < epu_diff_smooth) & (epu_diff.shift(1) >= epu_diff_smooth.shift(1))
        cross_up = (epu_diff > epu_diff_smooth) & (epu_diff.shift(1) <= epu_diff_smooth.shift(1))
        
        # 5. 信号生成
        # 恐慌见顶衰竭 -> 宏观不确定性消退 -> 安全资产(TLT)抛售 -> 看空 (-1.0)
        signal.loc[recent_extreme_high & cross_down] = -1.0
        
        # 自满见底反转 -> 宏观不确定性重燃 -> 资金涌入避险资产(TLT) -> 看多 (+1.0)
        signal.loc[recent_extreme_low & cross_up] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(diff_window={self.diff_window}, z_window={self.z_window}, smooth_window={self.smooth_window})"