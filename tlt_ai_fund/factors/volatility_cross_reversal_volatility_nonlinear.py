import numpy as np
import pandas as pd

class VolatilityCrossReversalFactor:
    """波动率极值与拥挤反转 (volatility/nonlinear)

    逻辑: 跨资产波动率(美股VIX + 黄金GVZ)的极度飙升代表流动性恐慌与对冲盘极度拥挤。当恐慌指标同步从极值开始回落时，标志着系统性恐慌耗竭(Panic Exhaustion)，流动性抛售压力解除，避险资金回流将引发美债脉冲式上涨；反之，当波动率处于极度低位且继续向下破位时，标志着风险狂热加速(Risk-on Melt-up)，避险资产遭抛售。采用严格衰竭条件确保不接飞刀，输出狙击手级脉冲信号。
    数据: vixcls, gvzcls
    触发: 
      - 看多: VIX 滚动Z-Score > 2.0 且 GVZ Z-Score > 1.5 (跨资产极值)，且满足衰竭条件(当日环比下降且低于3日均值)
      - 看空: VIX 与 GVZ 的 Z-Score < -1.5 (极度自满)，且继续向下破位
    输出: +1.0 或 -1.0 的脉冲信号，常态为 0.0
    """

    def __init__(self):
        self.name = 'volatility_cross_reversal_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (常态返回 0.0)
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'gvzcls']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        # 提取并前向填充数据，避免NaN阻断计算
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 1. 计算双窗口 Z-Score (捕捉中长期与短期的非线性极值特征)
        def calc_zscore(series, window):
            mean = series.rolling(window=window).mean()
            std = series.rolling(window=window).std().replace(0, np.nan)
            return (series - mean) / std

        vix_z252 = calc_zscore(vix, 252)
        vix_z63 = calc_zscore(vix, 63)
        gvz_z252 = calc_zscore(gvz, 252)
        gvz_z63 = calc_zscore(gvz, 63)
        
        # 2. 铁律2与3: 二阶导数与边际变化 (衰竭确认)
        vix_diff = vix.diff()
        gvz_diff = gvz.diff()
        
        # 衰竭条件: 当日环比下降 且 跌破短期均线
        vix_receding = (vix < vix.rolling(window=3).mean()) & (vix_diff < 0)
        gvz_receding = (gvz < gvz.rolling(window=3).mean()) & (gvz_diff < 0)
        
        # 3. 非线性交叉极值条件
        # 恐慌极值: VIX超过2.0，GVZ超过1.5 (多资产同频共振)
        panic_extreme = ((vix_z252 > 2.0) | (vix_z63 > 2.0)) & ((gvz_z252 > 1.5) | (gvz_z63 > 1.5))
        
        # 自满极值: VIX与GVZ均低于-1.5 (极度平缓)
        complacency_extreme = ((vix_z252 < -1.5) | (vix_z63 < -1.5)) & ((gvz_z252 < -1.5) | (gvz_z63 < -1.5))
        
        # 4. 生成信号
        # 看多脉冲: 极度恐慌 + 开始回落 -> 抄底债市 (Panic Exhaustion)
        long_cond = panic_extreme & vix_receding & gvz_receding
        
        # 看空脉冲: 极度自满 + 继续下破 -> 风险狂热加速，抛售债市 (Risk-on Melt-up)
        short_cond = complacency_extreme & vix_receding & gvz_receding
        
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"