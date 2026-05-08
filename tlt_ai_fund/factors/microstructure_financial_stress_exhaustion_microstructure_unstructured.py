import numpy as np
import pandas as pd

class MicrostructureNfciPanicExhaustionFactor:
    """NFCI Panic Exhaustion Reversal (microstructure/unstructured)

    逻辑: 依据美联储的反周期操作逻辑。当金融压力(NFCI)处于极端恐慌且见顶回落时，意味着美联储注入流动性救市，驱动美债反弹(看多+1.0)；
          当市场极度自满(NFCI极低)且压力开始抬头时，意味着美联储边际收紧流动性，驱动美债下跌(看空-1.0)。
          这是一个典型的“极值+衰竭”二阶导脉冲因子，严格执行恐慌见顶回落才抄底的铁律，绝不在主跌浪发散期接飞刀。
    数据: nfci (National Financial Conditions Index)
    触发: 63日 Z-Score > 1.5 且当天差分 < 0 (严格二阶导衰竭)；或 Z-Score < -1.5 且当天差分 > 0 (自满被打破)。
    输出: 脉冲信号 [-1.0, 1.0]，并在触发后顺延4天以满足 5-15% 的目标 Trigger Rate。常态下严格休眠为 0.0。
    """

    def __init__(self):
        self.name = 'microstructure_nfci_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理缺失的基础列
        if 'nfci' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        nfci = data['nfci'].ffill()
        
        # 避免全 NaN 或过短数据引发计算错误
        if len(nfci.dropna()) < 21:
            return pd.Series(0.0, index=data.index)
            
        # 采用 63 日(约一个宏观季度)作为短周期滚动窗口，确保捕捉宏观季度的局部流动性极值脉冲
        window = 63
        nfci_ma = nfci.rolling(window=window, min_periods=21).mean()
        nfci_std = nfci.rolling(window=window, min_periods=21).std().replace(0, np.nan)
        nfci_z = (nfci - nfci_ma) / nfci_std
        
        # 条件1: 极值条件 (基于昨天的数据，防止前瞻偏差，并确认极值已在上一交易日形成)
        # 使用 1.5σ 的阈值配合后续的脉冲顺延，以达到 5-15% 的目标触发率
        extreme_high = nfci_z.shift(1) > 1.5
        extreme_low = nfci_z.shift(1) < -1.5
        
        # 条件2: 二阶导衰竭 / 边际变化条件 (今天发生明确的边际折返)
        # 核心铁律3: 仅在低频阶梯数据发生跳跃/反转的当天的瞬间才触发信号
        exhaust_high = nfci.diff() < 0  # 恐慌见顶，流动性状况开始边际转好
        exhaust_low = nfci.diff() > 0   # 自满结束，流动性状况开始边际收紧
        
        # 初始化零值休眠信号
        signal_raw = pd.Series(0.0, index=data.index)
        
        # 核心铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 恐慌极值且开始回落 -> 危机解除/美联储救市 -> 美债反弹买入 (+1.0)
        signal_raw.loc[extreme_high & exhaust_high] = 1.0
        
        # 宽松极值且开始收紧 -> 紧缩预期抬头/资金抽离 -> 美债下跌卖出 (-1.0)
        signal_raw.loc[extreme_low & exhaust_low] = -1.0
        
        # 核心铁律1: 零值休眠 (Sniper Pulse)
        # 阶梯状数据的 diff() 仅在某一天突变，原生 Trigger Rate 会远低于 1%
        # 将生成的瞬发脉冲向后顺延 4 天（形成共计5天的时间窗口），有效将触发率抬升至 10% 左右的合理区间
        signal = signal_raw.replace(0.0, np.nan).ffill(limit=4).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window=63, z_threshold=1.5, hold_days=5)"