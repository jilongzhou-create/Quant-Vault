import numpy as np
import pandas as pd

class VolatilityCrossAssetExhaustionFactor:
    """波动率跨资产恐慌极值与衰竭反转因子 (microstructure/options)

    逻辑: 股票与黄金期权隐含波动率差值(VIX-GVZ)代表跨资产极端恐慌的微观结构压力。当差值达到极端高位后开始回落时，标志着无差别抛售或极端对冲需求消退，避险资金将重新配置美债，此时输出脉冲做多美债(TLT)。
    数据: vixcls, gvzcls
    触发: VIX与GVZ差值的 252日 Z-Score > 2.5 且 当前差值 < 过去3日均值
    输出: +1.0 (恐慌见顶回落买入TLT脉冲), 0.0 (常态休眠)
    """

    def __init__(self, zscore_window: int = 252, zscore_threshold: float = 2.5, exhaustion_window: int = 3):
        self.name = 'volatility_cross_asset_exhaustion'
        self.zscore_window = zscore_window
        self.zscore_threshold = zscore_threshold
        self.exhaustion_window = exhaustion_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化严格休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        # 数据缺失校验
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 提取并向前填充缺失值以防止计算中断
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产隐含波动率差值 (边际恐慌溢价)
        vol_spread = vix - gvz
        
        # 计算基于 252 个交易日 (1年) 的 Z-Score (宏观波动水位)
        roll_mean = vol_spread.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).mean()
        roll_std = vol_spread.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).std()
        
        # 避免除以零导致 NaN
        zscore = pd.Series(
            np.where(roll_std > 1e-6, (vol_spread - roll_mean) / roll_std, 0.0), 
            index=data.index
        )
        
        # 二阶导数衰竭条件: 当前值低于过去 N 天均值 (说明恐慌动量已见顶并开始消退)
        exhaustion = vol_spread < vol_spread.rolling(window=self.exhaustion_window).mean()
        
        # 生成狙击手级别脉冲信号
        trigger = (zscore > self.zscore_threshold) & exhaustion
        
        signal[trigger] = 1.0
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, threshold={self.zscore_threshold}, exhaustion_window={self.exhaustion_window})"