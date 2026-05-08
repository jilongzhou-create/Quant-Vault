import numpy as np
import pandas as pd

class EpuVolumePanicExhaustionFactor:
    """经济政策不确定性与微观放量衰竭因子 (microstructure/unstructured)

    逻辑: 结合非结构化新闻数据(EPU)的宏观恐慌极值与微观结构(成交量)的恐慌抛售。当经济政策不确定性飙升且TLT极端放量时，代表恐慌盘涌出；一旦不确定性回落且成交量萎缩(边际衰竭)，即触发极短期的抄底反弹脉冲，避免在暴跌主浪接飞刀。
    数据: usepuindxd (经济政策不确定性指数), volume (TLT成交量)
    触发: 过去5日内 usepuindxd 252日 Z-Score > 2.5 且 volume 63日 Z-Score > 2.0，当前 usepuindxd < 3日均值 且 成交量 < 昨日成交量 (恐慌衰竭)。
    输出: +1.0 看多美债(抄底)，其余时间为 0.0 (休眠)
    """

    def __init__(self, epu_z_window=252, vol_z_window=63, epu_z_threshold=2.5, vol_z_threshold=2.0):
        self.name = 'epu_volume_panic_exhaustion'
        self.epu_z_window = epu_z_window
        self.vol_z_window = vol_z_window
        self.epu_z_threshold = epu_z_threshold
        self.vol_z_threshold = vol_z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 处理数据缺失情况
        if 'usepuindxd' not in data.columns or 'volume' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        vol = data['volume'].ffill()
        
        # 1. 宏观不确定性极值 (非结构化数据边际跳跃)
        epu_mean = epu.rolling(window=self.epu_z_window, min_periods=self.epu_z_window//2).mean()
        epu_std = epu.rolling(window=self.epu_z_window, min_periods=self.epu_z_window//2).std()
        epu_zscore = (epu - epu_mean) / epu_std.replace(0, 1e-5)
        
        # 2. 微观恐慌放量极值 (微观结构)
        vol_mean = vol.rolling(window=self.vol_z_window, min_periods=self.vol_z_window//2).mean()
        vol_std = vol.rolling(window=self.vol_z_window, min_periods=self.vol_z_window//2).std()
        vol_zscore = (vol - vol_mean) / vol_std.replace(0, 1e-5)
        
        # 记录近期是否出现过极端恐慌共振
        extreme_panic = (epu_zscore > self.epu_z_threshold) & (vol_zscore > self.vol_z_threshold)
        recent_panic = extreme_panic.rolling(window=5, min_periods=1).max() > 0
        
        # 3. 衰竭反转条件 (二阶导数铁律: 禁止在指标处于绝对最高点时买入)
        # EPU 边际回落
        epu_exhaustion = epu < epu.rolling(window=3).mean()
        # 成交量边际萎缩，抛压减轻
        vol_exhaustion = vol < vol.shift(1)
        
        # 触发脉冲: 近期有过恐慌极值，当前已脱离最高点且满足双重衰竭
        buy_cond = recent_panic & epu_exhaustion & vol_exhaustion & (~extreme_panic)
        
        # 确保只在预期发生反转的瞬间脉冲触发 (边际变化铁律)
        trigger = buy_cond & (~buy_cond.shift(1).fillna(False))
        
        signal[trigger] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"EpuVolumePanicExhaustionFactor(epu_z_window={self.epu_z_window}, vol_z_window={self.vol_z_window}, epu_z_thresh={self.epu_z_threshold}, vol_z_thresh={self.vol_z_threshold})"