import numpy as np
import pandas as pd

class CrossAssetVolReversalFactor:
    """跨资产波动率极值反转因子 (microstructure/options)

    逻辑: 捕捉期权隐含波动率的微观结构反转。当VIX或股金波动率差值(VIX-GVZ)达到极端高位并开始回落时，标志着流动性恐慌极值衰竭，市场预期央行救市释放流动性，利好美债(看多)。相反，当VIX长期处于极端低位(极度自满)且开始抬头时，往往对应美联储收紧流动性带来的利率冲击预期(Hawkish Shock)，此时风险资产与长端美债往往面临双杀，利空美债(看空)。因子严格过滤连续状态，仅在反转瞬间输出脉冲。
    数据: vixcls, gvzcls
    触发: 
      看多(+1.0): VIX或VIX-GVZ 252日 Z-Score > 2.5 且 当日值 < 过去3日均值 (恐慌衰竭)
      看空(-1.0): VIX 252日 Z-Score < -2.0 且 当日值 > 过去3日均值 (自满衰竭/冲击起步)
    输出: 脉冲型信号，[-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'cross_asset_vol_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 初始状态全为 0.0，维持零值休眠
        signal = pd.Series(0.0, index=data.index)
        
        has_vix = 'vixcls' in data.columns
        has_gvz = 'gvzcls' in data.columns
        
        if not has_vix:
            return signal
            
        # 避免前视偏差
        vix = data['vixcls'].ffill()
        
        # 计算 VIX 252日 Z-Score
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        vix_zscore = (vix - vix_mean) / vix_std
        
        # 铁律2: 二阶导数，极度恐慌 + 开始回落
        vix_panic_exhaustion = (vix_zscore > 2.5) & (vix < vix.rolling(window=3).mean())
        
        # 铁律2: 二阶导数，极度自满 + 开始抬头 (边际变化)
        vix_complacency_exhaustion = (vix_zscore < -2.0) & (vix > vix.rolling(window=3).mean())
        
        long_cond = vix_panic_exhaustion
        
        if has_gvz:
            gvz = data['gvzcls'].ffill()
            vol_spread = vix - gvz
            
            # 计算 VIX与GVZ差值 的 252日 Z-Score
            spread_mean = vol_spread.rolling(window=252, min_periods=60).mean()
            spread_std = vol_spread.rolling(window=252, min_periods=60).std().replace(0, np.nan)
            spread_zscore = (vol_spread - spread_mean) / spread_std
            
            # 铁律2: 跨资产恐慌消退脉冲
            spread_panic_exhaustion = (spread_zscore > 2.5) & (vol_spread < vol_spread.rolling(window=3).mean())
            long_cond = long_cond | spread_panic_exhaustion
            
        # 赋值脉冲信号
        signal[long_cond] = 1.0
        signal[vix_complacency_exhaustion] = -1.0
        
        # 处理极端异常的冲突重叠，确保逻辑纯粹性
        conflict = long_cond & vix_complacency_exhaustion
        signal[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"