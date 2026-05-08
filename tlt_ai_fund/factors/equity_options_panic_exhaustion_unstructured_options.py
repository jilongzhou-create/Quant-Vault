import numpy as np
import pandas as pd

class VixRegimeReversalFactor:
    """波动微观结构突变 (unstructured/options)

    逻辑: 捕捉跨资产避险情绪(Risk-Off/On)的极端反转。股市恐慌情绪极值(VIX)往往指引债市的避险资金流向。当VIX极端高且出现动量衰退时，标志着避险情绪退潮(Risk-On)，资金从美债撤出，美债价格下跌，触发看空(-1.0)；相反，当VIX极度自满且突发跳涨时，标志着恐慌爆发(Risk-Off)，资金涌入安全的美国国债，美债价格上涨，触发看多(+1.0)。
    数据: vixcls (CBOE VIX 波动率指数)
    触发: 63日动态 Z-Score 极值 (Z>1.2 或 Z<-1.2) + 均线交叉 + 单日动量突变幅度 (二阶导回落/跳跃)。
    输出: [-1.0, 1.0] 的狙击级别脉冲信号。
    """

    def __init__(self):
        self.name = 'vix_regime_reversal'
        self.window = 63

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        
        # 处理有效数据
        if vix.dropna().empty:
            return signal

        # 计算短期（季度）局部动态统计量
        vix_mean = vix.rolling(window=self.window, min_periods=21).mean()
        vix_std = vix.rolling(window=self.window, min_periods=21).std()
        
        # 计算标准化的波动率水位 (Z-Score)
        vix_z = (vix - vix_mean) / (vix_std + 1e-6)
        
        # 计算波动率的一阶动量及动量的波动率
        vix_diff = vix.diff()
        vix_diff_std = vix_diff.rolling(window=self.window, min_periods=21).std()
        vix_ma3 = vix.rolling(window=3).mean()
        
        # 核心铁律：二阶导数衰竭（绝对禁止纯极值直接买卖）
        
        # 场景A: 恐慌衰竭 (Risk-On) -> 资金流出美债 -> 看空美债 (-1.0)
        # 1. 处于季度极端恐慌 (Z > 1.2)
        # 2. 开始回落并跌破近期均线 (vix < vix_ma3)
        # 3. 单日边际变化呈显著断层回落 (vix_diff < -0.25 * vix_diff_std)
        bearish_cond = (
            (vix_z > 1.2) & 
            (vix < vix_ma3) & 
            (vix_diff < -0.25 * vix_diff_std)
        )
        
        # 场景B: 自满打破 (Risk-Off) -> 资金涌入美债 -> 看多美债 (+1.0)
        # 1. 处于季度极度自满 (Z < -1.2)
        # 2. 突发危机突破近期均线 (vix > vix_ma3)
        # 3. 单日边际变化呈显著脉冲飙升 (vix_diff > 0.5 * vix_diff_std)
        bullish_cond = (
            (vix_z < -1.2) & 
            (vix > vix_ma3) & 
            (vix_diff > 0.5 * vix_diff_std)
        )
        
        # 赋值脉冲信号
        signal.loc[bearish_cond] = -1.0
        signal.loc[bullish_cond] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window})"