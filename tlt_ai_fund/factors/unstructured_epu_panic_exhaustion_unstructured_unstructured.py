import numpy as np
import pandas as pd

class UnstructuredEpuPanicExhaustionFactor:
    """政策不确定性恐慌极值衰竭因子 (unstructured/unstructured)

    逻辑: 经济政策不确定性(EPU)新闻指数(usepuindxd)反映了宏观层面的系统性政策恐慌。当EPU在短期内极度飙升(反映黑天鹅或政策休克)且动能开始衰竭时，标志着政策恐慌见顶(美联储往往被迫转鸽救市)，触发避险资金买盘，此时做多美债；反之，当EPU极度低迷且停止下降时，标志着市场极度自满，经济过热且央行具备充足的紧缩空间，此时做空美债。
    数据: usepuindxd (美国经济政策不确定性每日指数)
    触发: EPU 10日边际变化量的 252日 Z-Score > 2.5 且单日变化动能回落 -> +1.0; Z-Score < -2.5 且动能反弹 -> -1.0
    输出: 狙击手级脉冲信号，[-1.0, 1.0]
    """

    def __init__(self, momentum_window=10, z_window=252, z_threshold=2.5):
        self.name = 'unstructured_epu_panic_exhaustion'
        self.momentum_window = momentum_window
        self.z_window = z_window
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 核心铁律3: 边际变化 (Marginal Change Only)
        # EPU指数为基于新闻文本的非结构化数据，日频噪音极大，先用3日均线平滑
        epu = data['usepuindxd'].ffill()
        epu_smooth = epu.rolling(window=3, min_periods=1).mean()
        
        # 计算 10日(约两周) 边际变化量，捕捉短期内政策预期的突变脉冲
        epu_change = epu_smooth.diff(self.momentum_window)
        
        # 计算 252日(约一年) 滚动 Z-Score
        epu_change_mean = epu_change.rolling(window=self.z_window, min_periods=self.z_window//2).mean()
        epu_change_std = epu_change.rolling(window=self.z_window, min_periods=self.z_window//2).std()
        
        # 加上 1e-6 防止分母为 0
        epu_zscore = (epu_change - epu_change_mean) / (epu_change_std + 1e-6)
        
        # 核心铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 必须等待冲击到达极值且开始回落（动量衰竭）才触发信号，切忌在主升浪中逆势接飞刀
        epu_change_diff = epu_change.diff(1)
        
        # 触发条件设计 (核心铁律1: 零值休眠，控制极低触发率的脉冲):
        # 看多: 政策恐慌极度高涨 (Z-Score > 2.5) 且 恐慌边际放缓 (diff < 0) -> 避险资金涌入，美债上涨
        long_cond = (epu_zscore > self.z_threshold) & (epu_change_diff < 0)
        
        # 看空: 政策极度稳定自满 (Z-Score < -2.5) 且 自满边际结束 (diff > 0) -> 风险偏好极高，紧缩预期升温，美债下跌
        short_cond = (epu_zscore < -self.z_threshold) & (epu_change_diff > 0)
        
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"UnstructuredEpuPanicExhaustionFactor(momentum_window={self.momentum_window}, z_window={self.z_window}, z_threshold={self.z_threshold})"