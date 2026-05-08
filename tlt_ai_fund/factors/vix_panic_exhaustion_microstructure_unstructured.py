import numpy as np
import pandas as pd

class MicrostructureVixRegimePulseFactor:
    """微观结构/VIX不同状态极值与衰竭反转脉冲因子

    逻辑: VIX对美债(TLT)的影响是非线性的，常规线性因子会产生严重的内部冗余(导致Marginal Contribution不足)。
          本因子设计了三个正交的非线性脉冲反转状态:
          1. 恐慌极值衰竭 (Z>2.0且回落): 极致的Dash-for-cash流动性危机解除，资金重新回流美债避险，输出看多脉冲 (+1.0)
          2. 极度贪婪破裂 (Z<-1.2且回升): 市场长期麻痹后突然破防，典型的Risk-Off初潮，资金涌入美债，输出看多脉冲 (+1.0)
          3. 温和恐慌消退 (1.0<Z<1.5且回落): 常态化扰动结束，市场重归Goldilocks/Risk-On，资金流出美债，输出看空脉冲 (-1.0)
    数据: vixcls (VIX波动率指数)
    触发: 动态Z-Score分档 + 均值/动量二阶衰竭条件
    输出: [-1.0, 1.0] 的脉冲信号，常态绝对休眠 (0.0)，目标 Trigger Rate 控制在 8%~12% 之间。
    """

    def __init__(self):
        self.name = 'microstructure_vix_regime_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 严格遵守铁律1：常态下必须返回零值休眠的脉冲信号
        signal = pd.Series(0.0, index=data.index)
        
        # 异常处理：如果缺少核心字段，直接返回全0信号
        if 'vixcls' not in data.columns:
            return signal

        vix = data['vixcls'].ffill()

        # 计算 252日(约1年) 滚动 Z-Score，反映相对波动率水位
        roll_mean = vix.rolling(window=252, min_periods=63).mean()
        roll_std = vix.rolling(window=252, min_periods=63).std()
        
        # 避免除0警告
        roll_std = roll_std.replace(0, np.nan)
        vix_z = (vix - roll_mean) / roll_std

        # 严格遵守铁律3：边际变化与二阶导数 (Anti-Catch-Falling-Knife)
        # 计算动量与短期衰竭参考线
        vix_ma5 = vix.rolling(window=5).mean()
        vix_diff = vix.diff(1)

        # 状态1: 恐慌极值衰竭 (Liquidity Crisis Exhaustion)
        # 条件：水位极高(Z>2.0) 且 跌破5日均值 且 绝对值真实回落
        cond_panic_exhaustion = (vix_z > 2.0) & (vix < vix_ma5) & (vix_diff < 0)

        # 状态2: 极度贪婪破裂 (Complacency Broken / Flight to Safety)
        # 条件：水位极低(Z<-1.2) 且 突破5日均值 且 绝对值真实拉升
        cond_complacency_broken = (vix_z < -1.2) & (vix > vix_ma5) & (vix_diff > 0)

        # 状态3: 温和恐慌消退 (Moderate Fear Fading / Risk-On)
        # 条件：水位中高(1.0<Z<1.5) 且 跌破5日均值 且 绝对值真实回落
        # 说明：由于处于温和区间，VIX回落代表经济复苏、风险偏好回升，此时美债遭到抛售
        cond_moderate_fading = (vix_z > 1.0) & (vix_z < 1.5) & (vix < vix_ma5) & (vix_diff < 0)

        # 信号赋值 (各状态在数学上通过 Z-Score 阈值严格互斥，不会出现覆盖冲突)
        signal.loc[cond_panic_exhaustion] = 1.0
        signal.loc[cond_complacency_broken] = 1.0
        signal.loc[cond_moderate_fading] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"