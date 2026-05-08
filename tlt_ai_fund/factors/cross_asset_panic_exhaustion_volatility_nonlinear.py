import numpy as np
import pandas as pd

class CrossAssetPanicExhaustionFactor:
    """波动率极值与拥挤反转 (volatility/nonlinear)

    逻辑: 监控 VIX(美股) 和 GVZ(黄金) 的跨资产波动率狂飙。在流动性枯竭(无差别抛售)阶段避险资产亦被错杀(如2020年3月/2022年)；当双波动率处于极端高位且同步出现二阶导衰竭回落时，标志着强平潮瓦解，资金重返避险资产，输出反转做多美债脉冲信号。
    数据: vixcls, gvzcls
    触发: VIX 252日 Z-Score > 2.0 且 GVZ 252日 Z-Score > 1.5，同时两者当日差分 < 0 且低于3日均值
    输出: +1.0 表示恐慌抛售瓦解，做多美债(TLT)；常态非触发日严格为 0.0
    """

    def __init__(self):
        self.name = 'cross_asset_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 常态下严格返回全 0 的 Series
        signal = pd.Series(0.0, index=data.index)
        
        # 必须验证所需数据列是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal

        # 处理数据缺失 (使用前向填充保证对齐，若无新数据 diff() 将为 0，不会误触发衰竭)
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()

        # 计算 252日滚动 Z-Score (体现水位的极端程度)
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std()
        vix_z = (vix - vix_mean) / vix_std

        gvz_mean = gvz.rolling(window=252, min_periods=60).mean()
        gvz_std = gvz.rolling(window=252, min_periods=60).std()
        gvz_z = (gvz - gvz_mean) / gvz_std

        # 铁律2: 二阶导数防接飞刀 (极值条件 + 衰竭条件)
        # 条件1: 跨资产恐慌处于极端水位
        extreme_panic = (vix_z > 2.0) & (gvz_z > 1.5)
        
        # 条件2: 边际变化出现衰竭 (当日下降 且 跌破3日均线，确认情绪拐点)
        vix_exhaustion = (vix.diff() < 0) & (vix < vix.rolling(window=3).mean())
        gvz_exhaustion = (gvz.diff() < 0) & (gvz < gvz.rolling(window=3).mean())

        # 组合触发: 多重恐慌指标同步极值且同步衰竭
        trigger_cond = extreme_panic & vix_exhaustion & gvz_exhaustion
        
        # 脉冲输出: 仅在预期发生改变的衰竭瞬间触发 +1.0
        signal[trigger_cond] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"