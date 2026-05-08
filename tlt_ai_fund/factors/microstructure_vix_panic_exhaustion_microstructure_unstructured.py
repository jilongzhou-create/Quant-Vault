import numpy as np
import pandas as pd

class MicrostructureVixPanicExhaustionFactor:
    """微观结构波动率恐慌极值与衰竭反转因子 (Microstructure / Panic Exhaustion)

    逻辑: 捕捉流动性危机导致的无差别抛售恐慌极值。由于美债(TLT)在极致恐慌(VIX狂飙)阶段会被作为流动性资产遭到抛售，此时直接抄底会死于主跌浪。只有当VIX处于极端高位(Z-Score > 2.5)且开始回落(低于3日均值且单日下跌)时，才标志着恐慌见顶、流动性挤兑衰竭，此时美债将迎来确定性极强的反弹。
    数据: vixcls (VIX 波动率指数)
    触发: VIX的252日 Z-Score > 2.5 AND VIX < VIX.rolling(3).mean() AND VIX.diff() < 0
    输出: 极短期脉冲信号。满足衰竭抄底条件当天输出 +1.0，常态下严格休眠保持 0.0
    """

    def __init__(self):
        self.name = 'microstructure_vix_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号必须为全0，满足零值休眠铁律
        signal = pd.Series(0.0, index=data.index)

        # 检查依赖数据是否存在
        if 'vixcls' not in data.columns:
            return signal

        # 提取 VIX 数据，处理缺失值防断档
        vix = data['vixcls'].ffill()

        # 计算长周期 (252个交易日，约1年) 的动态 Z-Score
        vix_mean_252 = vix.rolling(window=252, min_periods=60).mean()
        vix_std_252 = vix.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        vix_zscore = (vix - vix_mean_252) / vix_std_252

        # 铁律2: 二阶导数条件 (Anti-Catch-Falling-Knife)
        # 必须等待动量衰竭：当前值跌破短期均线，且出现单日实质性回落
        vix_ma3 = vix.rolling(window=3, min_periods=1).mean()
        vix_diff = vix.diff()
        
        is_exhausting = (vix < vix_ma3) & (vix_diff < 0)

        # 条件1: VIX 处于极端高位
        is_extreme = vix_zscore > 2.5

        # 组合脉冲触发条件：极致恐慌 + 边际衰竭
        buy_pulse = is_extreme & is_exhausting

        # 输出脉冲信号 (+1.0 看多美债)
        signal.loc[buy_pulse] = 1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"