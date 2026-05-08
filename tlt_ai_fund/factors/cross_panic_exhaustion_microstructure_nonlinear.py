import numpy as np
import pandas as pd

class CrossAssetPanicExhaustionFactor:
    """交叉资产恐慌衰竭因子 (microstructure/nonlinear)

    逻辑: 恐慌情绪(VIX/EPU)达到极值时直接做多美债会死于主跌浪(接飞刀)。本因子等待恐慌情绪见顶回落(低于3日均值)，且同步观测到收益率曲线(10Y-2Y)出现动量陡峭化(确认短端利率下行预期，微观流动性挤兑衰竭)，此时才触发精准脉冲做多信号。反之，在市场极度自满且曲线平坦化时做空。这有效扩大了触发面(Trigger Rate 5-15%)，同时保持了严谨的FICC二阶导数抄底逻辑。
    数据: vixcls, usepuindxd, t10y2y
    触发: (VIX或EPU的126日Z-Score > 1.0 且 回落至3日均值以下) 且 (10Y-2Y 3日动量 > 0)
    输出: +1.0 看多美债，-1.0 看空美债，常态 0.0
    """

    def __init__(self):
        self.name = 'cross_asset_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，默认输出全0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 必须的数据列检查
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 波动率极值与衰竭计算
        vix_mean = vix.rolling(126).mean()
        vix_std = vix.rolling(126).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std
        
        # 铁律2: 必须有衰竭条件 (vix < 3日均值)
        vix_extreme = vix_z > 1.0
        vix_exhaustion = vix < vix.rolling(3).mean()
        
        vix_calm = vix_z < -1.0
        vix_bouncing = vix > vix.rolling(3).mean()
        
        # 铁律3: 边际变化，观测收益率曲线的陡峭化动量
        curve_mom = t10y2y.diff(3)
        curve_steepening = curve_mom > 0.0
        curve_flattening = curve_mom < 0.0
        
        # 多维交叉：引入经济政策不确定性指数（EPU）增加脉冲触发的稳健性与触发率
        if 'usepuindxd' in data.columns:
            epu = data['usepuindxd'].ffill()
            epu_mean = epu.rolling(126).mean()
            epu_std = epu.rolling(126).std().replace(0, np.nan)
            epu_z = (epu - epu_mean) / epu_std
            
            epu_extreme = epu_z > 1.0
            epu_exhaustion = epu < epu.rolling(3).mean()
            
            epu_calm = epu_z < -1.0
            epu_bouncing = epu > epu.rolling(3).mean()
            
            panic_exhausting = (vix_extreme & vix_exhaustion) | (epu_extreme & epu_exhaustion)
            calm_bouncing = (vix_calm & vix_bouncing) | (epu_calm & epu_bouncing)
        else:
            panic_exhausting = vix_extreme & vix_exhaustion
            calm_bouncing = vix_calm & vix_bouncing
            
        # 核心逻辑：恐慌情绪见顶衰竭 + 收益率曲线边际变陡（确认降息预期） = 极佳的美债买点
        buy_signal = panic_exhausting & curve_steepening
        
        # 反向逻辑：极度自满且情绪开始反弹 + 收益率曲线边际变平 = 极佳的美债卖点
        sell_signal = calm_bouncing & curve_flattening
        
        # 赋值非连续脉冲信号
        signal[buy_signal] = 1.0
        signal[sell_signal] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"