import numpy as np
import pandas as pd

class MicrostructureFundingSqueezeFactor:
    """Microstructure / Nonlinear Feature Crossing

    逻辑: 交叉 FICC 内部的底层资金挤兑指标 (DFF 与 3个月国库券利差) 与宏观恐慌指标 (VIX)。当短端避险情绪极端化 (资金疯狂涌入最安全的 T-bills 导致其收益率暴跌，DFF-DTB3利差飙升) 且伴随 VIX 极值时，说明发生系统性流动性冲击。只有当两个指标同步出现边际衰竭 (跌破3日均线)，才确认为抛售枯竭点，此时买入 TLT 捕捉脉冲反弹。纯脉冲信号避免在流动性危机主跌浪中接飞刀。
    数据: dff (联邦基金有效利率), dtb3 (3个月国库券利率), vixcls (VIX波动率指数)
    触发: 资金利差(DFF-DTB3) 252日 Z-Score > 2.5 且开始回落 + VIX Z-Score > 2.5 且开始回落
    输出: +1.0 表示多重流动性恐慌见顶衰竭，看多美债脉冲；常态为 0.0
    """

    def __init__(self):
        self.name = 'microstructure_funding_squeeze_exhaustion_cross'
        self.window = 252
        self.exhaustion_window = 3
        self.z_threshold = 2.5

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 校验所需字段是否存在
        required_cols = ['dff', 'dtb3', 'vixcls']
        for col in required_cols:
            if col not in data.columns:
                return signal

        # 提取数据并向前填充避免部分假期导致的数据缺失断层
        dff = data['dff'].ffill()
        dtb3 = data['dtb3'].ffill()
        vix = data['vixcls'].ffill()

        # 计算底层资金流动性挤压指标 (Unsecured Overnight vs Safe Haven 3M)
        # 避险情绪爆发时，资金买爆 DTB3 导致其收益率暴跌，利差急剧扩大
        squeeze = dff - dtb3

        # 计算 252 日滚动 Z-Score (反映极端偏离程度)
        squeeze_mean = squeeze.rolling(window=self.window).mean()
        squeeze_std = squeeze.rolling(window=self.window).std()
        # 避免除以0
        squeeze_z = (squeeze - squeeze_mean) / squeeze_std.replace(0, np.nan)

        vix_mean = vix.rolling(window=self.window).mean()
        vix_std = vix.rolling(window=self.window).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)

        # 二阶导数衰竭条件 (动量转弱，跌破过去3天均值，确认见顶回落)
        squeeze_exhaustion = squeeze < squeeze.rolling(window=self.exhaustion_window).mean()
        vix_exhaustion = vix < vix.rolling(window=self.exhaustion_window).mean()

        # 非线性特征交叉：极值与衰竭的共振
        extreme_panic = (squeeze_z > self.z_threshold) & (vix_z > self.z_threshold)
        sync_exhaustion = squeeze_exhaustion & vix_exhaustion

        # 狙击手脉冲触发：双重指标极端恶化后，同时出现边际改善
        long_condition = extreme_panic & sync_exhaustion

        signal.loc[long_condition] = 1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold})"