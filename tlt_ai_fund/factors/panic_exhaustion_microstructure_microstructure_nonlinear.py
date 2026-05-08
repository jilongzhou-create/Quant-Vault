import numpy as np
import pandas as pd

class PanicExhaustionMicrostructureFactor:
    """多重流动性恐慌衰竭因子 (Panic Exhaustion Reversal)

    逻辑: 结合宏观波动率(VIX)与微观流动性/金融压力(NFCI)的双重非线性交叉。
          绝对禁止在高波动时接飞刀！必须满足恐慌指标均处于极端高位(Z-Score>2.5)，
          且两者都同时出现边际衰竭(低于近期均值)，表明流动性挤兑和抛售枯竭，此时爆发性做多美债(TLT)。
    数据: vixcls, nfci
    触发: VIX 252日 Z-Score > 2.5 且低于3日均值 AND NFCI Z-Score > 2.0 且低于5日均值
    输出: +1.0 (狙击手级脉冲)，其余非触发日严格为 0.0
    """

    def __init__(self):
        self.name = 'panic_exhaustion_microstructure_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (Sniper Pulse)
        signal = pd.Series(0.0, index=data.index)

        # 校验所需数据缺失情况
        if 'vixcls' not in data.columns or 'nfci' not in data.columns:
            return signal

        vix = data['vixcls'].ffill()
        nfci = data['nfci'].ffill()

        # 计算 252日(一年期) Z-Score 以判定系统性极端水位
        vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std().replace(0, np.nan)
        nfci_z = (nfci - nfci.rolling(252).mean()) / nfci.rolling(252).std().replace(0, np.nan)

        # 铁律2 & 3: 二阶导数与边际变化条件 (防接飞刀，捕捉衰竭)
        # VIX 的衰竭：当前值拐头向下，回落至过去3天均值之下
        vix_exhaustion = vix < vix.rolling(3).mean()
        
        # NFCI 的衰竭：当前值回落至过去5天均值之下 (NFCI为周频数据，使用5天平滑以解决发布日错位带来的 diff() == 0 假象)
        nfci_exhaustion = nfci < nfci.rolling(5).mean()

        # 极端水位条件 (脉冲触发的前提)
        vix_extreme = vix_z > 2.5
        nfci_extreme = nfci_z > 2.0

        # 非线性交叉验证：多重恐慌指标同步满足极值且同步进入衰竭状态
        trigger = vix_extreme & vix_exhaustion & nfci_extreme & nfci_exhaustion

        # 满足全部苛刻条件时爆发脉冲做多信号
        signal[trigger] = 1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"