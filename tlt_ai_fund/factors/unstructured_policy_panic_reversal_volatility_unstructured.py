import numpy as np
import pandas as pd

class UnstructuredPolicyPanicReversalFactor:
    """Unstructured Policy Panic Reversal (volatility/unstructured)

    逻辑: 监控基于新闻媒体的经济政策不确定性(EPU)与跨资产波动率(VIX)的共振。当非结构化新闻引发的不确定性与市场恐慌极度狂飙并开始边际衰竭时，代表抛压耗尽、悬靴落地，避险/抄底资金重新涌入美债(脉冲看多)；当市场极度自满(双指标冰点)且开始反弹时，代表通胀/紧缩担忧萌芽，引发利率飙升(脉冲看空)。
    数据: usepuindxd (经济政策不确定性指数), vixcls (VIX波动率指数)
    触发: 多头 = EPU 252日 Z-Score > 2.5 且 VIX Z-Score > 1.5，同时两者当日值跌破3日均值(衰竭)；空头 = EPU Z-Score < -2.0 且 VIX Z-Score < -1.5，同时两者突破3日均值(抬头)。
    输出: +1.0 (极度恐慌衰竭看多), -1.0 (极度自满反转看空), 0.0 (常态休眠)
    """

    def __init__(self, z_long=2.5, z_short=-2.0, vix_z_long=1.5, vix_z_short=-1.5, window=252, smooth=3):
        self.name = 'unstructured_policy_panic_reversal'
        self.z_long = z_long
        self.z_short = z_short
        self.vix_z_long = vix_z_long
        self.vix_z_short = vix_z_short
        self.window = window
        self.smooth = smooth

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化常态零值休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        # 字段完整性检查
        if 'usepuindxd' not in data.columns or 'vixcls' not in data.columns:
            return signal
            
        # 缺失值前向填充，防止 NaN 干扰
        usepu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        
        # 计算 252 日滚动 Z-Score，规避零除
        usepu_mean = usepu.rolling(window=self.window, min_periods=self.window//2).mean()
        usepu_std = usepu.rolling(window=self.window, min_periods=self.window//2).std().replace(0, np.nan)
        usepu_z = (usepu - usepu_mean) / usepu_std
        
        vix_mean = vix.rolling(window=self.window, min_periods=self.window//2).mean()
        vix_std = vix.rolling(window=self.window, min_periods=self.window//2).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std
        
        # 计算 3 日均值，用于二阶导数/边际衰竭判定
        usepu_ma = usepu.rolling(window=self.smooth, min_periods=2).mean()
        vix_ma = vix.rolling(window=self.smooth, min_periods=2).mean()
        
        # 多头脉冲触发逻辑：极值恐慌 + 跨域确认 + 开始回落(衰竭)
        long_cond = (
            (usepu_z > self.z_long) & 
            (vix_z > self.vix_z_long) & 
            (usepu < usepu_ma) & 
            (vix < vix_ma)
        )
        
        # 空头脉冲触发逻辑：极度自满过热 + 跨域确认 + 开始抬头(反转)
        short_cond = (
            (usepu_z < self.z_short) & 
            (vix_z < self.vix_z_short) & 
            (usepu > usepu_ma) & 
            (vix > vix_ma)
        )
        
        # 仅在触发点赋值脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, smooth={self.smooth})"