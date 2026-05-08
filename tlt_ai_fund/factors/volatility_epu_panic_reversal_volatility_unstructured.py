import numpy as np
import pandas as pd

class VolatilityEpuPanicReversalFactor:
    """Volatility EPU Panic Reversal (volatility/unstructured)

    逻辑: 监控非结构化新闻指数(EPU)与VIX波动的共振。当政策不确定性与市场波动双双达到极端狂飙且开始瓦解时，捕捉美债作为避险/宽松预期的左侧反转；当极度自满打破时捕捉紧缩预期。因避免接飞刀，必须等极端情绪开始衰竭才触发脉冲。
    数据: vixcls, usepuindxd
    触发: VIX与EPU边际变化的252日Z-Score均处于极端(>1.0或<-1.0)，且同时向3日均值回归确认衰竭。
    输出: [-1.0, 1.0] 的极值衰竭反转脉冲，常态休眠返回 0.0。
    """

    def __init__(self):
        self.name = 'volatility_epu_panic_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号为全 0.0 (严格遵守铁律1: 零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns or 'usepuindxd' not in data.columns:
            return signal

        vix = data['vixcls'].ffill()
        epu = data['usepuindxd'].ffill()

        # 铁律3: 边际变化 (Marginal Change Only)
        # EPU 每日新闻噪音极大，使用5日平滑，并严格计算其5日边际变化量，禁止使用绝对水位
        epu_ma5 = epu.rolling(window=5).mean()
        epu_diff5 = epu_ma5.diff(5)
        
        # 计算 252日滚动 Z-Score 确定极端水位 (动态基准，适应不同宏观周期)
        vix_z = (vix - vix.rolling(window=252).mean()) / vix.rolling(window=252).std()
        epu_diff_z = (epu_diff5 - epu_diff5.rolling(window=252).mean()) / epu_diff5.rolling(window=252).std()

        # 铁律2: 二阶导数/衰竭确认 (Anti-Catch-Falling-Knife)
        # 绝对禁止在极值期间接飞刀，必须等价格跌破/突破3日均线才确认单边狂热/恐慌已瓦解
        vix_exhausted = vix < vix.rolling(window=3).mean()
        epu_exhausted = epu_ma5 < epu_ma5.rolling(window=3).mean()

        vix_reversing = vix > vix.rolling(window=3).mean()
        epu_reversing = epu_ma5 > epu_ma5.rolling(window=3).mean()

        # 多头脉冲 (+1.0)：VIX与EPU突升至极高（恐慌狂飙）且同时开始回落（避险情绪顶峰瓦解，倒逼宽松预期，驱动长端美债大涨）
        long_cond = (
            (vix_z > 1.0) & 
            (epu_diff_z > 1.0) & 
            vix_exhausted & 
            epu_exhausted
        )

        # 空头脉冲 (-1.0)：VIX与EPU极度低迷（极度自满）且同时开始抬头（平静被打破，风险溢价重估，流动性收紧预期打击美债）
        short_cond = (
            (vix_z < -1.0) & 
            (epu_diff_z < -1.0) & 
            vix_reversing & 
            epu_reversing
        )

        # 触发脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        # 处理异常值，确保干净输出
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"