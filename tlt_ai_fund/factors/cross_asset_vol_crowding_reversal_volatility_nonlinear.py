import numpy as np
import pandas as pd

class CrossAssetVolCrowdingReversalFactor:
    """跨资产波动率极值与拥挤反转 (volatility/nonlinear)

    逻辑: 波动率狂飙代表市场全面恐慌，但此时避险资产往往遭遇流动性抛售(接飞刀)。只有在恐慌指数(VIX/GVZ)达到极端高位（空头极度拥挤），并且跨资产(股/金)波动率同步开始衰竭回落时，才确认流动性冲击结束，美债将迎来强力修复和避险资金涌入的上涨脉冲。反之，当市场极度安逸(波动率双低)并突然被打破时，做空美债以防范通胀/加息反弹冲击。
    数据: vixcls, gvzcls, usepuindxd
    触发: (VIX或GVZ的252日Z-Score>2.5 或 叠加>1.5) 且 跨资产波动率同步回落(diff<0 且低于3日均线)
    输出: 脉冲信号，极端恐慌瓦解时输出 +1.0，极度安逸被打破时输出 -1.0
    """

    def __init__(self, z_window=252, ma_window=3):
        self.name = 'cross_asset_vol_crowding_reversal_nonlinear'
        self.z_window = z_window
        self.ma_window = ma_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 信号 (铁律1: 零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必须的数据列
        required_cols = ['vixcls', 'gvzcls', 'usepuindxd']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 填充前值，防止缺失影响计算
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        epu = data['usepuindxd'].ffill()
        
        # 计算 252 日长周期 Z-Score
        vix_z = (vix - vix.rolling(self.z_window).mean()) / vix.rolling(self.z_window).std()
        gvz_z = (gvz - gvz.rolling(self.z_window).mean()) / gvz.rolling(self.z_window).std()
        epu_z = (epu - epu.rolling(self.z_window).mean()) / epu.rolling(self.z_window).std()
        
        # 多头条件1: 跨资产恐慌极度拥挤 (单边极端或多重共振)
        extreme_panic = (vix_z > 2.5) | (gvz_z > 2.5) | ((vix_z > 1.5) & (gvz_z > 1.5)) | ((vix_z > 1.5) & (epu_z > 1.5))
        
        # 多头条件2: 跨资产恐慌同步瓦解 (铁律2: 二阶导数，避免接飞刀)
        panic_exhaustion = (vix.diff() < 0) & (gvz.diff() < 0) & (vix < vix.rolling(self.ma_window).mean())
        
        # 多头触发 (脉冲 +1.0)
        long_cond = extreme_panic & panic_exhaustion
        
        # 空头条件1: 市场极度安逸 (股金波动率双低)
        extreme_complacency = (vix_z < -1.5) & (gvz_z < -1.0)
        
        # 空头条件2: 安逸幻觉被打破 (边际波动率起跳)
        complacency_broken = (vix.diff() > 0) & (gvz.diff() > 0) & (vix > vix.rolling(self.ma_window).mean())
        
        # 空头触发 (脉冲 -1.0)
        short_cond = extreme_complacency & complacency_broken
        
        # 严格控制在触发瞬间输出非零值
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_window={self.z_window}, ma_window={self.ma_window})"