import numpy as np
import pandas as pd

class MacroPanicCrowdingReversalFactor:
    """宏观恐慌拥挤反转因子 (volatility/nonlinear)

    逻辑: 监控跨资产恐慌情绪的极端爆发。当股市(VIX)、黄金(GVZ)或政策不确定性(EPU)的252日Z-Score飙升至>2.5极端高位时，绝对禁止直接买入(防接飞刀)。必须等待多重波动率指标同步出现二阶衰竭(跌破3日均值且diff<0)，同时美债收益率曲线出现牛陡动量确认(t10y2y.diff>0)，才认定为流动性踩踏终结。此时输出看多脉冲并保持5天(满足Sniper频率)。反之，极度自满且波动率突发飙升时看空。
    数据: vixcls, gvzcls, usepuindxd, t10y2y
    触发: (Z-Score > 2.5) & (跨资产波动率同步回落) & (收益率曲线边际走陡) -> +1.0 脉冲
    输出: 严格狙击手级脉冲信号 [-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'macro_panic_crowding_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始信号严格设为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['vixcls', 'gvzcls', 'usepuindxd', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # 数据对齐与前向填充，防止非交易日引发的意外断层
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        epu = data['usepuindxd'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 1. 绝对水位 (第一层极值判断) - 252日滚动 Z-Score
        vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
        gvz_z = (gvz - gvz.rolling(252).mean()) / gvz.rolling(252).std()
        epu_z = (epu - epu.rolling(252).mean()) / epu.rolling(252).std()
        
        # 极度恐慌: 任一宏观/跨资产恐慌指标突破 2.5 标准差
        extreme_panic = (vix_z > 2.5) | (gvz_z > 2.5) | (epu_z > 2.5)
        
        # 极度自满: 波动率被压抑到历史低位
        extreme_complacency = (vix_z < -1.5) | (gvz_z < -1.5)
        
        # 2. 铁律2: 二阶导数确认 (Anti-Catch-Falling-Knife)
        # 绝对禁止纯粹因为 VIX>40 就买入，必须叠加动量衰竭 (diff < 0 且 跌破3日均线)
        vix_exhaustion = (vix.diff() < 0) & (vix < vix.rolling(3).mean())
        gvz_exhaustion = (gvz.diff() < 0) & (gvz < gvz.rolling(3).mean())
        
        vix_surge = (vix.diff() > 0) & (vix > vix.rolling(3).mean())
        gvz_surge = (gvz.diff() > 0) & (gvz > gvz.rolling(3).mean())
        
        # 3. 铁律3: 边际变化 (Marginal Change Only)
        # 引入收益率曲线作为跨资产的 FICC 核心确认，不看绝对水位，只看 3 日动量是否边际走陡/走平
        curve_steepening = t10y2y.diff(3) > 0
        curve_flattening = t10y2y.diff(3) < 0
        
        # 4. 非线性条件交叉组装
        # 多重极端 + 动量共振衰竭 + FICC 交叉确认
        long_trigger = extreme_panic & vix_exhaustion & gvz_exhaustion & curve_steepening
        short_trigger = extreme_complacency & vix_surge & gvz_surge & curve_flattening
        
        # 5. 脉冲信号展期 (Sniper Pulse 维持极短几天，保障 Trigger Rate 在 5%-15% 黄金区间)
        long_pulse = long_trigger.rolling(window=5, min_periods=1).max().fillna(0)
        short_pulse = short_trigger.rolling(window=5, min_periods=1).max().fillna(0)
        
        # 6. 赋值输出
        signal.loc[long_pulse == 1] = 1.0
        signal.loc[short_pulse == 1] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"