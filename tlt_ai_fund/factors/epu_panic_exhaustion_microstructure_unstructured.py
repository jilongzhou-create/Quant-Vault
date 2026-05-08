import numpy as np
import pandas as pd

class EpuPanicExhaustionFactor:
    """微观新闻情绪恐慌见顶脉冲因子 (microstructure/unstructured)

    逻辑: 每日经济政策不确定性指数(usepuindxd)代表了非结构化新闻文本中提取的微观恐慌情绪。
          当该指数飙升至252日极端高位(Z-Score > 2.5)且出现短期动量衰竭(低于3日均值)时，
          标志着政策恐慌见顶释放。此时流动性挤兑或通胀恐慌极值释放完毕，长端美债迎来绝佳抄底点。
    数据: usepuindxd (美国经济政策不确定性指数，非结构化文本衍生的高频数据)
    触发: 252日滚动 Z-Score > 2.5 (极值) 且 当日值 < 过去3日均值 (动量衰竭)
    输出: +1.0 (脉冲看多TLT), 常态 0.0
    """

    def __init__(self, z_threshold=2.5, window=252, exhaust_window=3):
        self.name = 'epu_panic_exhaustion_factor'
        self.z_threshold = z_threshold
        self.window = window
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 获取数据并前向填充，防缺失
        epu = data['usepuindxd'].ffill()
        
        # 建立历史基准 (一年期)
        epu_mean = epu.rolling(window=self.window).mean()
        epu_std = epu.rolling(window=self.window).std()
        
        # 避免除零错误
        epu_zscore = (epu - epu_mean) / (epu_std + 1e-8)
        
        # 短期动量基准 (判定是否开始回落)
        epu_short_mean = epu.rolling(window=self.exhaust_window).mean()
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 条件1: 处于极端恐慌水位
        condition_extreme = epu_zscore > self.z_threshold
        
        # 条件2: 恐慌开始边际衰竭 (二阶导数为负)
        condition_exhaustion = epu < epu_short_mean
        
        # 只有在极端恐慌爆发且确立见顶回落时，才触发极短期的狙击手脉冲信号
        trigger = condition_extreme & condition_exhaustion
        
        # 输出脉冲看多信号
        signal[trigger] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_threshold={self.z_threshold}, window={self.window}, exhaust_window={self.exhaust_window})"