import numpy as np
import pandas as pd

class EquityVolPanicExhaustionFactor:
    """恐慌极值与衰竭反转 (microstructure/options)

    逻辑: 捕捉流动性危机导致的恐慌抛售极值。在恐慌爆发极值阶段(VIX极端飙升)，往往伴随跨资产的"Dash for Cash"无差别抛售(包括作为避险资产的美债TLT也会遭到抛售)。当VIX动量衰竭(见顶回落)时，表明流动性危机初步解除，被错杀的避险资金将迅速重新涌入美债，此时触发高胜率的抄底做多脉冲。
    数据: vixcls (CBOE VIX 波动率指数)
    触发: 极度恐慌 (VIX 252日 Z-Score > 2.5) 且 动量衰竭 (VIX < 3日均值) -> 触发单日 +1.0 脉冲。
    输出: 严格的狙击手级极短期脉冲信号，常态下保持 0.0 休眠。
    """

    def __init__(self):
        self.name = 'equity_vol_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理数据缺失，返回全 0 休眠序列
        if 'vixcls' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        vix = data['vixcls'].ffill()
        
        # 1. 长期基准: 252日(约1个交易年)滚动分布，动态衡量波动率的极端水位
        vix_mean = vix.rolling(window=252, min_periods=126).mean()
        vix_std = vix.rolling(window=252, min_periods=126).std()
        vix_zscore = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        # 2. 短期动量: 3日均值，捕捉情绪边际回落的微观结构变化
        vix_3d_mean = vix.rolling(window=3, min_periods=1).mean()
        
        # 3. 核心铁律: 二阶导数 (Anti-Catch-Falling-Knife)
        # 条件1: 处于极端高位 (Z-Score > 2.5) -> 拒绝日常的普通波动
        # 条件2: 开始衰竭回落 (vix < 3日均值) -> 拒绝在主跌浪中接飞刀
        long_cond = (vix_zscore > 2.5) & (vix < vix_3d_mean)
        
        # 4. 组装初始信号
        raw_signal = pd.Series(0.0, index=data.index)
        raw_signal[long_cond] = 1.0
        
        # 5. 核心铁律: 边际变化与零值休眠 (Sniper Pulse)
        # 过滤掉极值区间的连续触发，只在"状态改变的第一天"(边际变化瞬间)开枪
        signal = pd.Series(0.0, index=data.index)
        
        # 触发脉冲：当前条件满足，且前一天不满足 (严格控制 Trigger Rate)
        trigger_pulse = (raw_signal == 1.0) & (raw_signal.shift(1) == 0.0)
        signal[trigger_pulse] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"