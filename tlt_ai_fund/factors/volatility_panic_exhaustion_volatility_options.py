import numpy as np
import pandas as pd

class VolatilityPanicExhaustionFactor:
    """Volatility Panic Exhaustion (volatility/options)

    逻辑: 恐慌抛售期间资金往往抽离一切资产（现金为王，包含抛售美债）。当跨资产期权隐含波动率狂飙至极端高位时，标志着无差别流动性冲击达到极值。一旦波动率开始边际衰竭（二阶导数向下），表明流动性恐慌解除，配置盘资金将重新涌入避险的长端美债，产生强烈的做多脉冲。反之，极端自满且波动率抬头时，预示着风险冲击再临，脉冲做空。
    数据: vixcls (美股隐含波动率), gvzcls (黄金隐含波动率，跨资产确认)
    触发: 恐慌衰竭做多脉冲: 63日 VIX Z-Score > 2.5 且 VIX/GVZ 同步跌破3日均线。自满反转做空脉冲: 63日 VIX Z-Score < -1.5 且 VIX/GVZ 同步突破3日均线。
    输出: +1.0 (看多美债), -1.0 (看空美债), 常态 0.0 (狙击手休眠)
    """

    def __init__(self):
        self.name = 'volatility_panic_exhaustion_options'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 63日(约一季度)滚动基准: 适应波动率聚集特性，保证适当脉冲触发率
        vix_mean = vix.rolling(63).mean()
        vix_std = vix.rolling(63).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std
        
        # 边际变化(二阶导数铁律): 绝对水位高/低本身不足以触发，必须用 3日均线 捕捉微观动量的反转衰竭
        vix_short_ma = vix.rolling(3).mean()
        gvz_short_ma = gvz.rolling(3).mean()
        
        # 狙击手条件1：极值恐慌 (Z > 2.5) + 跨资产恐慌同步衰竭 (价格 < 短期均线) -> 做多流动性修复 (TLT)
        panic_extreme = vix_z > 2.5
        panic_exhaustion = (vix < vix_short_ma) & (gvz < gvz_short_ma)
        long_cond = panic_extreme & panic_exhaustion
        
        # 狙击手条件2：极度自满 (鉴于VIX右偏对数正态分布特性，Z < -1.5 即可代表极度自满) + 跨资产恐慌抬头 -> 做空波动率冲击
        complacency_extreme = vix_z < -1.5
        complacency_reversal = (vix > vix_short_ma) & (gvz > gvz_short_ma)
        short_cond = complacency_extreme & complacency_reversal
        
        # 赋值脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"