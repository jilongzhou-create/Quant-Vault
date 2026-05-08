import numpy as np
import pandas as pd

class PanicCrossExhaustionFactor:
    """多重恐慌极值与衰竭交叉因子 (microstructure/nonlinear)

    逻辑: 结合股市恐慌(VIX)与宏观金融流动性压力(NFCI)。当多重恐慌指标同时达到极端高位时，代表跨资产级别的流动性危机（如2020年3月）。只有当两者同步出现回落(低于近期均值)时，才确认恐慌见顶衰竭，此时央行往往被迫释放流动性，美债(TLT)将迎来确定性的避险与宽松双重反弹。常态下输出0.0，严格遵守零值休眠铁律。
    数据: vixcls, nfci
    触发: (VIX 252日 Z-Score > 2.5 且 < 3日均值) AND (NFCI 252日 Z-Score > 2.0 且 < 5日均值)
    输出: 满足交叉衰竭条件时输出 +1.0 (狙击手级脉冲抄底)，非触发日严格为 0.0
    """

    def __init__(self, vix_z_threshold: float = 2.5, nfci_z_threshold: float = 2.0):
        self.name = 'panic_cross_exhaustion_microstructure_nonlinear'
        self.vix_z_threshold = vix_z_threshold
        self.nfci_z_threshold = nfci_z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 Series，严格遵守铁律1: 零值休眠
        signal = pd.Series(0.0, index=data.index)
        
        # 校验必须的数据列，缺失则返回全 0
        if 'vixcls' not in data.columns or 'nfci' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        nfci = data['nfci'].ffill()
        
        # 1. 计算 252日(约1交易年) 滚动 Z-Score，衡量指标是否处于宏观极端水位
        vix_mean = vix.rolling(window=252, min_periods=126).mean()
        vix_std = vix.rolling(window=252, min_periods=126).std().replace(0, 1e-6)
        vix_z = (vix - vix_mean) / vix_std
        
        nfci_mean = nfci.rolling(window=252, min_periods=126).mean()
        nfci_std = nfci.rolling(window=252, min_periods=126).std().replace(0, 1e-6)
        nfci_z = (nfci - nfci_mean) / nfci_std
        
        # 2. 铁律2 & 3: 二阶导数与边际变化 (动量衰竭条件)
        # 绝对禁止在高位直接买入(接飞刀)，必须等待动量开始反转
        # VIX 变动极快，使用 3日均值判断衰竭；
        # NFCI 是周频阶梯状数据(在此处为前填)，使用 5日(1周)均值捕获边际改善瞬间
        vix_exhaustion = vix < vix.rolling(window=3, min_periods=1).mean()
        nfci_exhaustion = nfci < nfci.rolling(window=5, min_periods=1).mean()
        
        # 3. 非线性特征交叉触发逻辑
        # 股市极度恐慌 + 流动性极度紧缩 + 两者同时出现动量衰竭
        long_condition = (
            (vix_z > self.vix_z_threshold) & 
            (nfci_z > self.nfci_z_threshold) & 
            vix_exhaustion & 
            nfci_exhaustion
        )
        
        # 只在触发当天给出 +1.0 信号
        signal.loc[long_condition] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(vix_z_threshold={self.vix_z_threshold}, nfci_z_threshold={self.nfci_z_threshold})"