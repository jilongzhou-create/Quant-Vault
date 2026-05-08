import numpy as np
import pandas as pd

class CrossAssetVolPanicPulseFactor:
    """恐慌共振衰竭因子 (panic_mean_reversion/nonlinear)

    逻辑: 当风险资产(VIX)与避险资产(GVZ, 黄金波动率)同时飙升至极值, 表明市场进入无差别抛售的流动性恐慌; 一旦VIX见顶回落(二阶导数为负), 即为极佳抄底脉冲; 而当VIX缓慢上升但尚未极端时为钝刀割肉, 看空。
    数据: vixcls, gvzcls
    输出: +1.0 强烈看多(恐慌共振且初现衰竭), -1.0 看空(阴跌升波阶段), 0.0 常态休眠
    触发条件: 做多: VIX Z-Score>1.8且GVZ Z-Score>1.0且今日VIX回落低于3日均线; 做空: VIX Z-Score处于[0.8, 1.8]且连涨两日。预期Trigger Rate约8%。
    """

    def __init__(self):
        self.name = 'cross_asset_vol_panic_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 缺失关键列则直接返回0.0
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算 252日 Z-Score 衡量各自的历史极端程度
        vix_roll = vix.rolling(window=252)
        gvz_roll = gvz.rolling(window=252)
        
        vix_z = (vix - vix_roll.mean()) / vix_roll.std()
        gvz_z = (gvz - gvz_roll.mean()) / gvz_roll.std()
        
        # 边际变化(动量与二阶导数)
        vix_diff = vix.diff()
        vix_diff_prev = vix.shift(1).diff()
        vix_ma3 = vix.rolling(window=3).mean()
        
        # 强看多脉冲 (+1.0)：跨资产波动率极度飙升(流动性危机)，且今日VIX开始衰竭
        long_cond = (
            (vix_z > 1.8) & 
            (gvz_z > 1.0) & 
            (vix_diff < 0) & 
            (vix < vix_ma3)
        )
        
        # 看空脉冲 (-1.0)：钝刀割肉，VIX持续缓慢上升但未到极值，避险未出现共振恐慌(假摔阶段)
        short_cond = (
            (vix_z > 0.8) & 
            (vix_z <= 1.8) & 
            (vix_diff > 0) & 
            (vix_diff_prev > 0) & 
            (gvz_z < 1.0)
        )
        
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"