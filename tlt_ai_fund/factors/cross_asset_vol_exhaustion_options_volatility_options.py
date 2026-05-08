import numpy as np
import pandas as pd

class CrossAssetVolExhaustionOptionsFactor:
    """波动率极值与拥挤反转 (volatility/options)

    逻辑: 股市与黄金期权隐含波动率(VIX/GVZ)同时极端狂飙代表无差别的流动性危机抛售(包含美债被错杀)。当双波动率自历史极值开始回落(二阶导衰竭)时，恐慌与流动性挤兑消退，避险资金重返美债，触发看多脉冲；反之，长期极低波动率代表拥挤自满，其被打破(波动率突然抬头)时往往预示通胀担忧或流动性收紧，触发看空脉冲。常态下零值休眠。
    数据: vixcls, gvzcls
    触发: VIX Z-Score > 2.5 且 GVZ Z-Score > 2.0 加上 跌破3日均值及前值(衰竭) -> +1.0；双Z-Score极低且开始抬头 -> -1.0
    输出: [-1.0, 1.0] 脉冲信号
    """

    def __init__(self, window=252, z_long_vix=2.5, z_long_gvz=2.0, z_short=-1.5):
        self.name = 'cross_asset_vol_exhaustion_options'
        self.window = window
        self.z_long_vix = z_long_vix
        self.z_long_gvz = z_long_gvz
        self.z_short = z_short

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 必须检查数据完整性，若缺少则返回全 0 信号
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()

        # 计算 252 日滚动 Z-Score 以衡量水位的极端程度
        vix_mean = vix.rolling(self.window).mean()
        vix_std = vix.rolling(self.window).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std

        gvz_mean = gvz.rolling(self.window).mean()
        gvz_std = gvz.rolling(self.window).std().replace(0, np.nan)
        gvz_z = (gvz - gvz_mean) / gvz_std

        # 条件1: 极度恐慌 + 衰竭 -> 看多 TLT (+1.0) (遵守二阶导数铁律，必须叠加衰竭条件)
        # 极高且开始回落，确认危机解除与挤兑消退
        extreme_panic = (vix_z > self.z_long_vix) & (gvz_z > self.z_long_gvz)
        vix_exhaustion = (vix < vix.rolling(3).mean()) & (vix.diff() < 0)
        gvz_exhaustion = (gvz < gvz.rolling(3).mean()) & (gvz.diff() < 0)
        
        buy_pulse = extreme_panic & vix_exhaustion & gvz_exhaustion

        # 条件2: 极端自满 + 打破 -> 看空 TLT (-1.0) (边际变化铁律)
        # 极低水平突然抬头，标志着拥挤的宏观平稳期被打破，引发抛售
        extreme_complacency = (vix_z < self.z_short) & (gvz_z < self.z_short)
        vix_spike = (vix > vix.rolling(3).mean()) & (vix.diff() > 0)
        gvz_spike = (gvz > gvz.rolling(3).mean()) & (gvz.diff() > 0)
        
        short_pulse = extreme_complacency & vix_spike & gvz_spike

        # 赋值狙击手脉冲信号，常态保持休眠为 0.0
        signal[buy_pulse] = 1.0
        signal[short_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window})"