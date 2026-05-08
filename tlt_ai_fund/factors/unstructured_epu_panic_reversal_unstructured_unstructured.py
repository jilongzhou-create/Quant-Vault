import numpy as np
import pandas as pd

class UnstructuredEpuPanicReversalFactor:
    """经济政策不确定性(EPU)见顶反转脉冲因子 (Unstructured/Unstructured)

    逻辑: 经济政策不确定性指数(usepuindxd)基于新闻文本分析。当不确定性极其剧烈飙升时，暗示重大宏观恐慌冲击，这通常迫使央行转向宽松预期以救市；但为避免在流动性危机初期的无差别抛售中接飞刀，必须等待 EPU 动能见顶且开始边际回落的瞬间，才触发美债买入脉冲。
    数据: usepuindxd (美国经济政策不确定性指数，非结构化数据转化)
    触发: EPU 20日边际变化量的 252日 Z-Score > 2.5 (极度恐慌) + EPU 3日变化 < 0 (二阶导数衰竭)
    输出: 脉冲信号 +1.0 (不确定性见顶，避险资金涌入美债) / -1.0 (不确定性极度出清后反转，避险退潮)
    """

    def __init__(self, roc_window=20, z_window=252, exhaust_window=3, z_threshold=2.5):
        self.name = 'unstructured_epu_panic_reversal'
        self.roc_window = roc_window
        self.z_window = z_window
        self.exhaust_window = exhaust_window
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        
        # 降噪：非结构化新闻数据日频噪音大，采用5日均线轻度平滑以提取真实趋势
        epu_smooth = epu.rolling(window=5, min_periods=1).mean()
        
        # 核心铁律3: 边际变化 (关注不确定性的累积跃升幅度)
        epu_chg = epu_smooth.diff(self.roc_window)
        
        # 滚动 Z-Score 衡量事件的极端性
        roll_mean = epu_chg.rolling(window=self.z_window, min_periods=self.z_window // 2).mean()
        roll_std = epu_chg.rolling(window=self.z_window, min_periods=self.z_window // 2).std()
        roll_std = roll_std.replace(0, np.nan)
        
        z_score = (epu_chg - roll_mean) / roll_std
        
        # 核心铁律2: 二阶导数 (Anti-Catch-Falling-Knife 衰竭条件)
        # 必须等待动量开始反转才能介入
        exhaustion_long = epu_smooth.diff(self.exhaust_window) < 0
        exhaustion_short = epu_smooth.diff(self.exhaust_window) > 0
        
        # 触发条件逻辑
        # 多头：不确定性剧烈飙升（Z > 2.5）且开始回落（衰竭） -> 避险买盘，政策被迫转鸽 -> 做多TLT
        long_cond = (z_score > self.z_threshold) & exhaustion_long
        
        # 空头：不确定性罕见地大幅出清（Z < -2.5，如软着陆确认）且企稳反弹 -> 避险退潮，重回通胀/紧缩逻辑 -> 做空TLT
        short_cond = (z_score < -self.z_threshold) & exhaustion_short
        
        # 核心铁律1: 零值休眠 (Sniper Pulse)
        # 通过剔除连续触发的冗余信号，保证只有预转折的瞬间输出脉冲
        long_pulse = long_cond & (~long_cond.shift(1).fillna(False))
        short_pulse = short_cond & (~short_cond.shift(1).fillna(False))
        
        signal.loc[long_pulse] = 1.0
        signal.loc[short_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(roc_window={self.roc_window}, z_window={self.z_window}, exhaust_window={self.exhaust_window}, z_threshold={self.z_threshold})"