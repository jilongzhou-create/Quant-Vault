import numpy as np
import pandas as pd

class EpuExhaustionPulseFactor:
    """经济政策不确定性脉冲因子 (policy_pivot/unstructured)

    逻辑: 经济政策不确定性(EPU)基于新闻NLP生成。它的飙升代表市场面临突发政策冲击(轻微恐慌), 风险偏好恶化, 看空; 
          当不确定性极度悲观并触顶回落时, 政策明朗化, 风险溢价大幅压缩, 提供抄底极佳买点(极值+衰竭)。
    数据: [usepuindxd] (基于新闻文本分析的美国经济政策不确定性日度指数)
    输出: +1.0(抄底看多), -1.0(恶化看空), 0.0(常态休眠)
    触发条件: 滚动Z-Score>0.8且动量发生负偏转触发+1.0; 平静期(<0.5)突然发生>1.25倍标准差跳跃触发-1.0 (Trigger Rate: 5%-15%)
    """

    def __init__(self):
        self.name = 'epu_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 提取EPU指数并前向填充避免缺失
        epu = data['usepuindxd'].ffill()
        
        # EPU为新闻衍生的高频噪音数据, 5日均线平滑提取主趋势
        epu_smooth = epu.rolling(window=5, min_periods=1).mean()
        
        # 计算1年期(约252交易日)相对水位 (Z-Score)
        roll_mean = epu_smooth.rolling(window=252, min_periods=60).mean()
        roll_std = epu_smooth.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        z_epu = ((epu_smooth - roll_mean) / roll_std).fillna(0.0)
        
        # 计算3日动量及其滚动标准化, 捕捉边际变化跳跃
        epu_mom = epu_smooth.diff(3)
        mom_std = epu_mom.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        z_epu_mom = (epu_mom / mom_std).fillna(0.0)
        
        # --- 抄底看多 (+1.0): 不确定性极值 + 衰竭回落 ---
        # 1. 之前处于高不确定性区域 (历史前约20%)
        elevated_epu = z_epu.shift(1) > 0.8
        # 2. 不确定性开始实质性消退 (动量<0)
        falling_epu = epu_mom < 0.0
        # 3. 狙击手脉冲: 捕捉刚好发生转折回落的瞬态
        just_started_falling = falling_epu & (~falling_epu.shift(1).fillna(False))
        buy_cond = elevated_epu & just_started_falling
        
        # --- 趋势恶化 (-1.0): 平静期突发政策黑天鹅 ---
        # 1. 前期相对平静 (<0.5)
        calm_epu = z_epu.shift(3) < 0.5
        # 2. 突发政策不确定性急剧飙升 (>1.25 倍标准差跳升)
        spike_epu = z_epu_mom > 1.25
        # 3. 狙击手脉冲: 捕捉刚跳升的瞬态
        just_spiked = spike_epu & (~spike_epu.shift(1).fillna(False))
        sell_cond = calm_epu & just_spiked
        
        # 写入脉冲信号
        signal.loc[sell_cond] = -1.0
        signal.loc[buy_cond] = 1.0  # 若同时触发(极小概率), 多头逻辑优先覆盖
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"