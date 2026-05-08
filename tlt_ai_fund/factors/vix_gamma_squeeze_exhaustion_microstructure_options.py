import numpy as np
import pandas as pd

class OptionsVolContagionExhaustionFactor:
    """波动率传染衰竭因子 (microstructure/options)

    逻辑: 捕捉期权市场极端恐慌与极度自满的边际反转。当VIX极端飙升且开始回落时, 标志股市流动性无差别抛售(Cash is King)结束, 美债避险属性重新主导, 收益率下行(看多); 当黄金波动率(GVZ)极端低迷且开始反弹时, 标志实际利率平淡期结束, 利率重估引发债市抛售(看空)。
    数据: vixcls, gvzcls
    触发: VIX 63日 Z-Score > 1.5 且跌破3日均值触发看多; GVZ 63日 Z-Score < -1.5 且突破3日均值触发看空。
    输出: 严格脉冲型 +1.0 或 -1.0, 常态为 0.0。
    """

    def __init__(self):
        self.name = 'options_vol_contagion_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需数据列是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算 63 个交易日(约一季度)的滚动均值和标准差，定义短期宏观波动率状态
        vix_mean = vix.rolling(window=63, min_periods=21).mean()
        vix_std = vix.rolling(window=63, min_periods=21).std()
        
        gvz_mean = gvz.rolling(window=63, min_periods=21).mean()
        gvz_std = gvz.rolling(window=63, min_periods=21).std()
        
        # 计算局部 Z-Score，捕捉短期的边际极端脉冲
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan).fillna(1e-5)
        gvz_z = (gvz - gvz_mean) / gvz_std.replace(0, np.nan).fillna(1e-5)
        
        # 二阶导数衰竭条件: 极值后的边际反转 (Anti-Catch-Falling-Knife)
        # 恐慌衰竭: 波动率日内回落且跌破3日均线
        vix_falling = (vix.diff() < 0) & (vix < vix.rolling(window=3).mean())
        
        # 自满觉醒: 波动率日内反弹且突破3日均线
        gvz_rising = (gvz.diff() > 0) & (gvz > gvz.rolling(window=3).mean())
        
        # 触发看多脉冲: 股市极度恐慌见顶回落 -> 避险资金回流美债 (规避主跌浪)
        long_cond = (vix_z > 1.5) & vix_falling
        
        # 触发看空脉冲: 实际利率极度自满被打破 -> 利率重新定价引发抛售
        short_cond = (gvz_z < -1.5) & gvz_rising
        
        # 赋值信号 (常态保持 0.0，仅脉冲瞬间赋予 +1/-1)
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"