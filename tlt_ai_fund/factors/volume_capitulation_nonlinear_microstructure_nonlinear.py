import numpy as np
import pandas as pd

class VolumeCapitulationNonlinearFactor:
    """TLT Volume Capitulation & Panic Exhaustion (microstructure/nonlinear)

    逻辑: 结合TLT微观成交量激增(极度抛售导致的流动性易手)与宏观恐慌指标(VIX)的高维交叉。当且仅当微观成交量极度放大且宏观VIX处于极端恐慌状态，随后两者同时出现回落(空头抛压耗尽、恐慌情绪退潮)时，确认为狙击级抄底脉冲，避免在主跌浪接飞刀。
    数据: volume (TLT微观成交量), vixcls (VIX波动率指数)
    触发: volume 252日 Z-Score > 1.5 且 vixcls 252日 Z-Score > 2.0，同时两者当日值均低于过去3日均值(严格衰竭条件)
    输出: +1.0 (多头脉冲，看多美债)
    """

    def __init__(self, vol_z_threshold=1.5, vix_z_threshold=2.0, window=252, exhaust_window=3):
        self.name = 'microstructure_volume_capitulation_nonlinear'
        self.vol_z_threshold = vol_z_threshold
        self.vix_z_threshold = vix_z_threshold
        self.window = window
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)

        # 检查所需数据列是否存在
        required_cols = ['volume', 'vixcls']
        for col in required_cols:
            if col not in data.columns:
                return signal

        # 获取数据并前向填充避免缺失值造成的计算跳跃
        volume = data['volume'].ffill()
        vix = data['vixcls'].ffill()

        # 计算微观成交量的长期 Z-Score (识别爆量抛售极值)
        vol_mean = volume.rolling(window=self.window, min_periods=60).mean()
        vol_std = volume.rolling(window=self.window, min_periods=60).std()
        vol_z = (volume - vol_mean) / (vol_std + 1e-8)  # 避免除以0

        # 计算宏观波动率的长期 Z-Score (识别宏观恐慌极值)
        vix_mean = vix.rolling(window=self.window, min_periods=60).mean()
        vix_std = vix.rolling(window=self.window, min_periods=60).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)

        # 铁律2 & 3: 二阶导数与边际变化 (动能衰竭过滤，防止接飞刀)
        # 当日值必须低于近期移动平均，代表动能已经拐头向下
        vol_exhaustion = volume < volume.rolling(window=self.exhaust_window).mean()
        vix_exhaustion = vix < vix.rolling(window=self.exhaust_window).mean()

        # 核心逻辑交叉组合: 极值触发 + 同步衰竭反转
        trigger = (
            (vol_z > self.vol_z_threshold) & 
            (vix_z > self.vix_z_threshold) & 
            vol_exhaustion & 
            vix_exhaustion
        )

        # 仅在触发极值衰竭的脉冲日输出 +1.0
        signal.loc[trigger] = 1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(vol_z={self.vol_z_threshold}, vix_z={self.vix_z_threshold}, window={self.window})"