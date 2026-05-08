import numpy as np
import pandas as pd

class FiccSafeHavenVolExhaustionFactor:
    """避险期权波动率衰竭因子 (volatility/options)

    逻辑: 黄金隐含波动率(GVZ)是FICC领域度量通胀与法币危机的核心期权指标。当GVZ或VIX极度飙升时, 标志着宏观避险恐慌达到极点; 只有当波动率从极端极值(Z-Score>2.5)出现二阶回落特征(低于3日均值且当日下降)时, 才确认流动性挤兑或通胀恐慌耗尽, 形成安全抄底美债(TLT)的狙击脉冲。反之, 极度自满且波动率觉醒时看空美债。
    数据: gvzcls, vixcls
    触发: 多头=任一波动率252日Z-Score>2.5 且 两者同步一阶下降并低于3日均值; 空头=任一波动率Z-Score<-2.0 且 同步上升并高于3日均值。
    输出: 脉冲信号, +1.0表示恐慌衰竭看多TLT, -1.0表示自满觉醒看空TLT。
    """

    def __init__(self):
        self.name = 'ficc_safe_haven_vol_exhaustion'
        self.window = 252
        self.z_high = 2.5
        self.z_low = -2.0

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 常态下信号必须返回 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 必须处理数据缺失的情况
        if 'gvzcls' not in data.columns or 'vixcls' not in data.columns:
            return signal
            
        gvz = data['gvzcls'].ffill()
        vix = data['vixcls'].ffill()
        
        # 计算 252 日滚动 Z-Score (纯粹自己领域数据)
        gvz_mean = gvz.rolling(window=self.window).mean()
        gvz_std = gvz.rolling(window=self.window).std()
        gvz_z = (gvz - gvz_mean) / gvz_std
        
        vix_mean = vix.rolling(window=self.window).mean()
        vix_std = vix.rolling(window=self.window).std()
        vix_z = (vix - vix_mean) / vix_std
        
        # 铁律3: 计算边际变化与 3 日均值
        gvz_3d_ma = gvz.rolling(window=3).mean()
        vix_3d_ma = vix.rolling(window=3).mean()
        
        gvz_diff = gvz.diff()
        vix_diff = vix.diff()
        
        # 多头条件：铁律2 (极值 + 衰竭)
        # 条件1: 跨资产波动率处于极端高位
        panic_extreme = (gvz_z > self.z_high) | (vix_z > self.z_high)
        # 条件2: 跨资产波动率同步回落确认衰竭
        exhausting = (gvz_diff < 0) & (vix_diff < 0) & (gvz < gvz_3d_ma) & (vix < vix_3d_ma)
        long_trigger = panic_extreme & exhausting
        
        # 空头条件：极度自满 + 边际觉醒 (反向逻辑)
        # 条件1: 跨资产波动率处于极端低位
        complacency_extreme = (gvz_z < self.z_low) | (vix_z < self.z_low)
        # 条件2: 跨资产波动率同步上升确认觉醒
        waking = (gvz_diff > 0) & (vix_diff > 0) & (gvz > gvz_3d_ma) & (vix > vix_3d_ma)
        short_trigger = complacency_extreme & waking
        
        # 零值休眠触发：只在极端事件发生瞬间赋值
        signal[long_trigger] = 1.0
        signal[short_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"