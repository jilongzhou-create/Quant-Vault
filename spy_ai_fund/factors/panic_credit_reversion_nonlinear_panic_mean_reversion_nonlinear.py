import numpy as np
import pandas as pd

class PanicCreditReversionNonlinearFactor:
    """恐慌均值回归非线性因子 (panic_mean_reversion/nonlinear)

    逻辑: 捕捉极端恐慌见顶回落的确定性买点，以及轻度恐慌爆发时的看空趋势。买点必须等待VIX与信用利差的极值与衰竭共振。
    数据: [vixcls, bamlh0a0hym2]
    输出: 强看多(+1.0)代表恐慌衰竭抄底，强看空(-1.0)代表恐慌发酵趋势恶化
    触发条件: VIX或信用利差Z-Score>1.5且开始回落则看多；Z-Score在0.5至2.5且加速飙升则看空。预期Trigger Rate 5%-15%。
    """

    def __init__(self):
        self.name = 'panic_credit_reversion_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['vixcls', 'bamlh0a0hym2']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        vix = data['vixcls'].ffill()
        hy = data['bamlh0a0hym2'].ffill()
        
        # 使用 252 日滚动窗口计算 Z-Score (代表市场隐含的恐慌和实际违约预期的极端程度)
        vix_roll = vix.rolling(window=252, min_periods=60)
        vix_z = (vix - vix_roll.mean()) / (vix_roll.std() + 1e-6)
        
        hy_roll = hy.rolling(window=252, min_periods=60)
        hy_z = (hy - hy_roll.mean()) / (hy_roll.std() + 1e-6)
        
        # 计算短期动量变化 (用于二阶导数识别)
        vix_diff1 = vix.diff(1)
        vix_diff3 = vix.diff(3)
        hy_diff1 = hy.diff(1)
        hy_diff3 = hy.diff(3)
        
        # 多头脉冲：极度恐慌 + 衰竭
        # 经济学含义：极度恐慌状态(高Z-Score)发生，但必须等待波动率和信用利差同步停止恶化并回落(二阶导数为负)
        is_panic_extreme = (vix_z > 1.5) | (hy_z > 1.5)
        is_exhausted = (vix_diff1 < 0) & (vix_diff3 < 0) & (hy_diff1 <= 0)
        long_signal = is_panic_extreme & is_exhausted
        
        # 空头脉冲：恐慌升温 + 趋势恶化
        # 经济学含义：正常平静状态被打破，恐慌情绪正在发酵，且VIX和信用利差呈现同步走阔加速的飞刀状态
        is_panic_rising = (vix_z > 0.5) & (vix_z <= 2.5)
        is_worsening = (vix_diff1 > 0) & (vix_diff3 > 1.5) & (hy_diff3 > 0.05)
        short_signal = is_panic_rising & is_worsening & (~long_signal)
        
        signal[long_signal] = 1.0
        signal[short_signal] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"