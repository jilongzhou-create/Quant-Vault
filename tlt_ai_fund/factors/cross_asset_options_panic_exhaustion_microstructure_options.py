import numpy as np
import pandas as pd

class CrossAssetOptionsPanicExhaustionFactor:
    """期权跨资产恐慌衰竭因子 (microstructure/options)

    逻辑: 结合美股期权隐波(VIX)和黄金期权隐波(GVZ)构建跨资产流动性恐慌指数。两者同时飙升代表发生无差别流动性危机，该指数见顶回落标志着恐慌衰竭和央行宽松预期发酵，引发美债脉冲式暴涨；反之，从极度自满的低位向上反弹意味着紧缩冲击，引发抛售。严格保持零值休眠，仅在预期逆转瞬间输出脉冲。
    数据: [vixcls, gvzcls]
    触发: 极值高位(合成Z-Score > 3.0) + 开始回落(diff < 0 且小于3日均值) -> +1.0；极度低位(合成Z-Score < -2.0) + 开始反弹 -> -1.0
    输出: [-1.0, 1.0] 的二阶导数脉冲信号
    """

    def __init__(self):
        self.name = 'cross_asset_options_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        window = 252
        
        vix_std = vix.rolling(window).std().replace(0, 1e-5)
        vix_z = (vix - vix.rolling(window).mean()) / vix_std
        
        gvz_std = gvz.rolling(window).std().replace(0, 1e-5)
        gvz_z = (gvz - gvz.rolling(window).mean()) / gvz_std
        
        panic_idx = vix_z + gvz_z
        
        extreme_panic = panic_idx > 3.0
        panic_exhaustion = (panic_idx < panic_idx.rolling(3).mean()) & (panic_idx.diff() < 0)
        
        extreme_complacency = panic_idx < -2.0
        complacency_reversal = (panic_idx > panic_idx.rolling(3).mean()) & (panic_idx.diff() > 0)
        
        signal[extreme_panic & panic_exhaustion] = 1.0
        signal[extreme_complacency & complacency_reversal] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"