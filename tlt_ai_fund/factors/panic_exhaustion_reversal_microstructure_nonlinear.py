import numpy as np
import pandas as pd

class PanicExhaustionReversalFactor:
    """恐慌极值与衰竭反转 (microstructure / nonlinear)

    逻辑: 在微观结构上的流动性危机极值处捕捉衰竭反转脉冲信号。当期权隐含波动率(VIX)或底层金融压力(NFCI)在短期内冲破极端高位时，表明流动性恐慌和无差别抛售已被充分计价。此时严格要求指标的二阶导数开始转负(同步边际回落且不出现相互背离)，表明系统性恐慌见顶，流动性开始修复。此时作为避险与流动性海绵的美债(TLT)将迎来极佳的反弹修复。这种反弹具有极短期的高动能，故严格做脉冲信号处理，非触发日休眠。
    数据: vixcls, nfci
    触发: VIX 63日 Z-Score > 2.5 且开始回落 (当日 < 3日均值) 
          OR 金融压力 NFCI 63日 Z-Score > 2.0 且开始边际改善。
          同时施加非线性交叉底线: 绝不接飞刀，两端指标在脉冲日必须均不处于恶化(飙升)状态。
    输出: +1.0 表示多重恐慌指标同步见顶衰竭，短期看多美债(TLT)反弹；其余时间严格为 0.0。
    """

    def __init__(self):
        self.name = 'panic_exhaustion_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns or 'nfci' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        nfci = data['nfci'].ffill()
        
        # 铁律3: 边际变化 (捕捉变化瞬间)
        # NFCI 为低频周度阶梯状发布数据，为避免阶梯陷阱，采用 5 日差分观测周度边际变化
        vix_diff = vix.diff()
        nfci_diff5 = nfci.diff(5)
        
        # 计算季度级别 (63日) 微观结构与压力的短周期 Z-Score 极值
        vix_z63 = (vix - vix.rolling(63).mean()) / vix.rolling(63).std()
        nfci_z63 = (nfci - nfci.rolling(63).mean()) / nfci.rolling(63).std()
        
        # 铁律2: 二阶导数 (衰竭判定)
        # 不仅当前动能下行，绝对值还必须跌破近期(3日/5日)均线，双重确认下跌惯性
        vix_exhaustion = (vix_diff < 0) & (vix < vix.rolling(3).mean())
        nfci_exhaustion = (nfci_diff5 < 0) & (nfci < nfci.rolling(5).mean())
        
        # 极端高位 + 衰竭
        vix_panic_exhausted = (vix_z63 > 2.5) & vix_exhaustion
        nfci_panic_exhausted = (nfci_z63 > 2.0) & nfci_exhaustion
        
        # 非线性交叉过滤防接飞刀:
        # 当波动率释放衰竭买入信号时，严禁金融压力端还在飙升(反之亦然)。
        # 两者在脉冲日必须同步呈现缓和或持平状态 (diff <= 0)。
        no_falling_knife = (vix_diff <= 0) & (nfci_diff5 <= 0)
        
        # 触发脉冲: 至少一侧达到极值衰竭，且整体系统处于修复协同中
        trigger = (vix_panic_exhausted | nfci_panic_exhausted) & no_falling_knife
        
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 只在多重极值与衰竭共振的节点进行单日点射买入
        signal[trigger] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"