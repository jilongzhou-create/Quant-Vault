import numpy as np
import pandas as pd

class OptionsVolPanicExhaustionFactor:
    """期权波动率恐慌突变与衰竭因子 (unstructured/options)

    逻辑: 捕捉期权市场(VIX)隐含波动率短期急剧飙升所代表的极端恐慌与流动性冲击。常态下保持 0.0 休眠避免摩擦。当VIX动量出现极端正向脉冲（一年一遇级恐慌）且随后出现日环比回落（二阶导数<0，代表恐慌动能与抛压衰竭）时，意味着避险资金与宽松预期将重新主导长端美债，脉冲做多(TLT)；反之极度贪婪并反转时做空。
    数据: vixcls (CBOE VIX 波动率指数)
    触发: VIX 5日变化量的 252日滚动 Z-Score > 2.5 且 今日VIX回落 -> +1.0 看多；Z-Score < -2.0 且 今日VIX反弹 -> -1.0 看空。
    输出: 仅在极端事件反转瞬间触发的脉冲信号 [-1.0, 1.0]。
    """

    def __init__(self):
        self.name = 'options_vol_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号必须为 pd.Series(0.0) 保证非触发日零值休眠
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        
        # 铁律3: 边际变化 - 绝对禁止使用 VIX 的绝对水位，使用 5日(一周)动量捕捉短期内的恐慌爆发
        vix_mom = vix.diff(5)
        
        # 使用 252日 (一整个自然年) 计算滚动均值和标准差，提供年度视角的基准
        roll_mean = vix_mom.rolling(window=252, min_periods=63).mean()
        roll_std = vix_mom.rolling(window=252, min_periods=63).std()
        
        # 计算动量的 Z-Score，防分母为0
        vix_mom_z = (vix_mom - roll_mean) / (roll_std + 1e-6)
        
        # 铁律2: 二阶导数防接飞刀 (极值 + 衰竭)
        
        # 极端恐慌(多美债)条件: 动量向上异动 > 2.5倍标准差
        extreme_panic = vix_mom_z > 2.5
        # 恐慌衰竭条件: VIX 本身的日环比开始回落 (代表流动性抛售结束，重回避险逻辑)
        panic_exhaustion = vix.diff(1) < 0
        
        # 极端贪婪(空美债)条件: 动量极度向下探底 < -2.0倍标准差 (防范常态波动，要求较高的极值)
        extreme_complacency = vix_mom_z < -2.0
        # 贪婪衰竭条件: VIX 触底后开始日环比反弹 (风险偏好急剧升温转移债市资金)
        complacency_exhaustion = vix.diff(1) > 0
        
        # 铁律1: 零值休眠 (仅在双条件同时满足的脉冲瞬间触发)
        buy_cond = extreme_panic & panic_exhaustion
        sell_cond = extreme_complacency & complacency_exhaustion
        
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"