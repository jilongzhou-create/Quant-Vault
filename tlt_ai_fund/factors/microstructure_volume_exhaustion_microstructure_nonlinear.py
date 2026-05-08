import numpy as np
import pandas as pd

class MicrostructureVolumeExhaustionFactor:
    """Microstructure Volume Exhaustion (microstructure/nonlinear)

    逻辑: 结合 TLT 微观交易量(volume)与宏观恐慌指数(vixcls)进行非线性特征交叉。由于 TLT 也是避险资产，在极端的流动性危机中(如2020年3月)，TLT 同样会被无差别抛售换取现金，导致微观成交量与宏观 VIX 同步爆表。只有当两个指标的乘积进入极端状态且恐慌开始衰竭时，才产生高胜率的抄底脉冲；反之，当极度贪婪且放量赶顶时，抓取看空脉冲。
    数据: volume (TLT微观成交量), vixcls (宏观波动率指数)
    触发: 交易量与 VIX 的同向交叉乘积 > 1.5 (非线性极值)，且当期 VIX 边际回落/抬头 (二阶导数条件)。
    输出: 脉冲信号，+1.0为恐慌衰竭抄底，-1.0为贪婪赶顶做空，其余时间严守零值休眠。
    """

    def __init__(self):
        self.name = 'microstructure_volume_exhaustion_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值休眠序列
        signal = pd.Series(0.0, index=data.index)
        
        # 容错处理：确保基础数据列存在
        if 'volume' not in data.columns or 'vixcls' not in data.columns:
            return signal
            
        vol = data['volume'].ffill()
        vix = data['vixcls'].ffill()
        
        # 1. 计算滚动 Z-Score (避免魔法数字，252对应年化交易日，63对应单季度初始化)
        vol_std = vol.rolling(window=252, min_periods=63).std().replace(0, np.nan)
        vix_std = vix.rolling(window=252, min_periods=63).std().replace(0, np.nan)
        
        vol_z = (vol - vol.rolling(window=252, min_periods=63).mean()) / vol_std
        vix_z = (vix - vix.rolling(window=252, min_periods=63).mean()) / vix_std
        
        # 2. 非线性特征交叉 (Nonlinear Feature Cross)
        # 恐慌交叉: VIX 偏高 且 成交量异常放大 -> 宏观与微观共振的流动性挤兑
        cross_panic = (vix_z * vol_z).where((vix_z > 0) & (vol_z > 0), 0.0)
        
        # 贪婪交叉: VIX 偏低(极度乐观) 且 成交量异常放大 -> 微观吹泡泡放量赶顶
        cross_complacency = ((-vix_z) * vol_z).where((vix_z < 0) & (vol_z > 0), 0.0)
        
        # 3. 衰竭与边际变化条件 (二阶导数防飞刀铁律)
        # 恐慌衰竭: 极值过后，VIX 当天环比下降，且已经跌破近3日均值
        panic_exhausting = (vix.diff() < 0) & (vix < vix.rolling(3).mean())
        
        # 贪婪衰竭: 赶顶过后，VIX 当天环比抬头，且已经升破近3日均值
        complacency_exhausting = (vix.diff() > 0) & (vix > vix.rolling(3).mean())
        
        # 4. 触发极少数日子的脉冲信号
        # 设置交叉乘积阈值为 1.5 (约对应两个维度的 Z-Score 同步超过 1.22 倍标准差)
        long_pulse = (cross_panic > 1.5) & panic_exhausting
        short_pulse = (cross_complacency > 1.5) & complacency_exhausting
        
        signal[long_pulse] = 1.0
        signal[short_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"