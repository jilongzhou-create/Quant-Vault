import numpy as np
import pandas as pd

class OptionsCrossAssetPanicExhaustionFactor:
    """跨资产恐慌消退脉冲 (unstructured/options)

    逻辑: 波动率微观结构中, VIX(美股隐含波动率)与GVZ(黄金隐含波动率)的差值代表跨资产流动性恐慌程度。当差值极端走高(流动性危机、无差别抛售), 随后开始回落时, 标志着美联储流动性注入或恐慌见顶, 资金重新配置进入长端美债(TLT)避险。
    数据: vixcls, gvzcls
    触发: 差值的252日 Z-Score > 2.5 且 差值 < 3日均值 (恐慌衰竭, 看多美债)；差值 Z-Score < -2.5 且 差值 > 3日均值 (自满破裂, 看空美债)。
    输出: 脉冲信号, +1.0 表示流动性恐慌衰竭(看多美债), -1.0 表示极度自满破裂(看空美债), 常态为 0.0。
    """

    def __init__(self):
        self.name = 'options_cross_asset_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'gvzcls']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 填充缺失值并获取序列
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产波动率差值 (Volatility Spread)
        vol_spread = vix - gvz
        
        # 计算 252 个交易日 (约一年) 的滚动 Z-Score
        roll_mean = vol_spread.rolling(window=252, min_periods=60).mean()
        roll_std = vol_spread.rolling(window=252, min_periods=60).std()
        
        # 增加微小偏置防止除零异常
        zscore = (vol_spread - roll_mean) / (roll_std + 1e-6)
        
        # 计算3日均值用于二阶导数衰竭判断
        spread_ma3 = vol_spread.rolling(window=3, min_periods=1).mean()
        
        # 极值 + 衰竭逻辑 (Anti-Catch-Falling-Knife)
        # 看多条件：恐慌极值 (Z > 2.5) 且恐慌开始回落 (当前差值 < 3日均值)
        long_condition = (zscore > 2.5) & (vol_spread < spread_ma3)
        
        # 看空条件：自满极值 (Z < -2.5) 且自满破裂 (当前差值 > 3日均值)
        short_condition = (zscore < -2.5) & (vol_spread > spread_ma3)
        
        signal[long_condition] = 1.0
        signal[short_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"