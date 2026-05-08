import numpy as np
import pandas as pd

class PolicyGoldVolExhaustionFactor:
    """政策与黄金期权波动共振衰竭因子 (unstructured/options)

    逻辑: 结合非结构化数据(经济政策不确定性指数EPU)与期权隐含波动率(黄金ETF隐含波动率GVZ)。在极端的宏观恐慌中，两者会产生共振飙升，暗示市场处于流动性危机或极端避险状态。这种极端状态往往迫使央行放鸽救市。当黄金波动率停止恶化并跌破3日均值时，确认恐慌动量衰竭，政策底出现，触发脉冲做多美债(TLT)捕捉宽松预期兑现的主升浪。反之亦然。
    数据: usepuindxd (EPU), gvzcls (GVZ)
    触发: (EPU 5日变化Z-Score + GVZ 5日变化Z-Score) > 2.5 且 GVZ < 3日均值 -> +1.0; 综合 Z-Score < -2.5 且 GVZ > 3日均值 -> -1.0
    输出: +1.0 看多美债(恐慌消退/宽松兑现), -1.0 看空美债(极端平稳被打破/紧缩预期发酵)
    """

    def __init__(self):
        self.name = 'policy_gold_vol_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)，禁止使用绝对水位，计算 5 个交易日的动量
        epu_diff = epu.diff(5)
        gvz_diff = gvz.diff(5)
        
        # 计算 252 日 (约1年) 的滚动 Z-Score，捕捉罕见的突发变盘
        epu_z = (epu_diff - epu_diff.rolling(252).mean()) / epu_diff.rolling(252).std()
        gvz_z = (gvz_diff - gvz_diff.rolling(252).mean()) / gvz_diff.rolling(252).std()
        
        # 构建非结构化政策冲击与期权隐含波动率的共振指数
        shock_index = epu_z + gvz_z
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 多头衰竭确认: 极端恐慌发生后，黄金期权波动率开始回落，意味着真正的央行干预生效或抛售枯竭
        gvz_falling = gvz < gvz.rolling(3).mean()
        # 空头反转确认: 极端死寂发生后，黄金期权波动率开始抬头，意味着太平盛世被打破
        gvz_rising = gvz > gvz.rolling(3).mean()
        
        # 铁律1: 零值休眠 (Sniper Pulse)，极端突变发生且伴随衰竭时才扣动扳机
        buy_cond = (shock_index > 2.5) & gvz_falling
        sell_cond = (shock_index < -2.5) & gvz_rising
        
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"