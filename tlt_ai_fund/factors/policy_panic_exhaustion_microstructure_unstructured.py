import numpy as np
import pandas as pd

class StlfsiPanicExhaustionPulseFactor:
    """StlfsiPanicExhaustionPulseFactor (microstructure/unstructured)

    逻辑: 捕捉金融系统压力的极值与衰竭反转(Panic Exhaustion Reversal)。
          极端高压衰竭：当圣路易斯联储金融压力指数(stlfsi4)飙升至极端高位(Z>1.5)，随后见顶回落时，标志着流动性危机(现金为王导致的抛售)衰竭，美联储救市生效，避险资金重新涌入美债，触发看多脉冲(+1.0)。
          极端自满衰竭：当金融压力处于极度低位(Z<-1.5，过度乐观)且开始反弹时，标志着宽松周期结束，流动性边际收紧(如加息预期升温)，触发看空美债脉冲(-1.0)。
    数据: stlfsi4 (St. Louis Fed Financial Stress Index)
    触发: 前一日的 252日 Z-Score > 1.5 且 当前值 < 3日均线 -> +1.0；前一日 Z-Score < -1.5 且 当前值 > 3日均线 -> -1.0。
    输出: [-1.0, 1.0] 脉冲信号，维持5天以保证 5-15% 的 Trigger Rate。
    """

    def __init__(self):
        self.name = 'stlfsi_panic_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 安全检查：确保所需字段存在
        if 'stlfsi4' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 前向填充周频数据为日频，确保数据连续且无前瞻偏差
        fsi = data['stlfsi4'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 计算 252个交易日(约1年)的滚动 Z-Score，捕捉压力的极端水位
        roll_mean = fsi.rolling(window=252, min_periods=63).mean()
        roll_std = fsi.rolling(window=252, min_periods=63).std().replace(0, np.nan)
        z_score = (fsi - roll_mean) / roll_std
        
        # 取前一日的 Z-Score，确保反转判定发生时，前序状态是极端水位
        prev_z = z_score.shift(1)
        
        # 铁律2: 二阶导数与衰竭 (Anti-Catch-Falling-Knife)
        # 衰竭条件：当周公布的新值与过去3天(上一周期的平移值)比较发生阶梯反转
        ma_3 = fsi.rolling(window=3).mean()
        
        # 恐慌极值 + 衰竭回落 -> 危机见顶，流动性恢复，做多美债
        pulse_long = ((prev_z > 1.5) & (fsi < ma_3)).astype(int)
        
        # 极度自满 + 压力反弹 -> 宽松红利耗尽，流动性边际收紧，做空美债
        pulse_short = ((prev_z < -1.5) & (fsi > ma_3)).astype(int)
        
        # 铁律1: 零值休眠与狙击手脉冲
        # 为了解决前次挖掘 Trigger rate 接近 0% 的问题，放宽了 Z阈值(1.5)，并将脉冲维持 5 天
        # 这确保了每次周频数据的边际变化能覆盖后续的一周交易日，使触发率稳定落在 5% - 15% 区间
        hold_long = pulse_long.rolling(window=5, min_periods=1).max()
        hold_short = pulse_short.rolling(window=5, min_periods=1).max()
        
        # 初始化严格的 0.0 序列
        signal = pd.Series(0.0, index=data.index)
        signal[hold_long == 1] = 1.0
        
        # 处理潜在冲突(极小概率同一天多空同时触发，由空头逻辑覆盖)
        signal[hold_short == 1] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"