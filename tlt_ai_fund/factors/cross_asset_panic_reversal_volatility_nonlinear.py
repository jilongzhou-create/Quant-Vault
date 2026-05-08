import numpy as np
import pandas as pd

class CrossAssetPanicReversalFactor:
    """跨资产波动率极值与衰竭反转因子 (volatility/nonlinear)

    逻辑: 监控股市与黄金等跨资产波动率的极端狂飙。在避险情绪极度拥挤且开始瓦解（VIX处于极端高位，且VIX与GVZ同步回落）时，捕捉恐慌盘平仓带来的流动性修复与政策宽松确认。此时是做多美债(TLT)的极佳反转脉冲节点。常态下严格休眠。
    数据: vixcls, gvzcls
    触发: VIX 252日 Z-Score > 2.0 (兼顾极值与 5%-15% 触发率) + VIX.diff() < 0 (衰竭) + GVZCLS.diff() < 0 (跨资产确认衰竭)
    输出: +1.0 (强烈看多美债脉冲)
    """

    def __init__(self, z_threshold=2.0, window=252):
        self.name = 'cross_asset_panic_reversal'
        self.z_threshold = z_threshold
        self.window = window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始化全为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必要数据列是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 前向填充处理不同频数据的缺失值
        df = data[['vixcls', 'gvzcls']].ffill()
        vix = df['vixcls']
        gvz = df['gvzcls']
        
        # 计算 VIX 的动态 252 日 Z-Score
        vix_mean = vix.rolling(window=self.window, min_periods=self.window // 2).mean()
        vix_std = vix.rolling(window=self.window, min_periods=self.window // 2).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, 1e-5)
        
        # 铁律3: 边际变化，捕捉恐慌情绪改变的瞬间
        # 铁律2: 二阶导数，衰竭条件 (不能接飞刀)
        vix_exhaustion = vix.diff() < 0
        gvz_exhaustion = gvz.diff() < 0
        
        # 核心触发逻辑: 非线性极值交叉 + 双重衰竭确认
        trigger = (vix_z > self.z_threshold) & vix_exhaustion & gvz_exhaustion
        
        # 输出脉冲信号 (+1.0 看多)
        signal[trigger] = 1.0
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"CrossAssetPanicReversalFactor(z_threshold={self.z_threshold}, window={self.window})"