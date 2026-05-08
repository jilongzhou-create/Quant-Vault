import numpy as np
import pandas as pd

class NewsEpuFlightToSafetyFactor:
    """新闻政策不确定性避险脉冲因子 (microstructure/unstructured)

    逻辑: 追踪基于非结构化新闻文本计算的经济政策不确定性指数(USEPUINDXD)。当新闻文本中爆发极端的政策不确定性黑天鹅（短期增量Z-Score > 2.5）时，将引发微观流动性层面的恐慌与资金避险抢筹（Flight to Quality/Safety），生成被动做多美债（TLT）的抄底脉冲。
    数据: [usepuindxd]
    输出: [+1.0 看多美债脉冲，非触发日 0.0]
    """

    def __init__(self, window=252, diff_days=3, z_threshold=2.5):
        self.name = 'news_epu_flight_to_safety'
        self.window = window
        self.diff_days = diff_days
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理数据缺失情况
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 获取数据并防止前瞻偏差
        epu = data['usepuindxd'].ffill()
        
        # 计算极短期的非结构化文本不确定性异动
        epu_diff = epu.diff(self.diff_days)
        
        # 基于约一年(252个交易日)的滚动基准刻画宏观常态，计算 Z-Score
        roll_mean = epu_diff.rolling(window=self.window, min_periods=self.window // 2).mean()
        roll_std = epu_diff.rolling(window=self.window, min_periods=self.window // 2).std()
        
        # 防止除零错误
        z_score = (epu_diff - roll_mean) / (roll_std + 1e-8)
        
        # 极端恐慌异动触发多头脉冲: 新闻不确定性极值飙升导致风险资产流动性黑洞，机构涌入美债
        signal = pd.Series(0.0, index=data.index)
        signal[z_score > self.z_threshold] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, diff_days={self.diff_days}, z_threshold={self.z_threshold})"