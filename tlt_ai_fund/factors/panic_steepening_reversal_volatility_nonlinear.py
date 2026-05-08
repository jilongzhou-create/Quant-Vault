import numpy as np
import pandas as pd

class VolatilityCrowdingReversalNonlinearFactor:
    """Volatility Crowding Reversal & Nonlinear Regime Factor (volatility/nonlinear)

    逻辑: 股债相关性在不同波动率极值下呈非线性变化, 因子根据 VIX 划分为三大脉冲政权:
         1. 极度恐慌衰竭 (VIX>32或Z>2.0): 流动性危机引发无差别抛售, 此时VIX回落意味着美联储救市(QE/降息), 强力看多美债。
         2. 普通避险衰竭 (1.0<Z<=2.0): 股市普通回调结束, 避险资金流出美债重返股市, 此时VIX回落应看空美债 (避免接飞刀)。
         3. 极度自满破裂 (Z<-1.5): 市场极度乐观后突发异动, 避险情绪初起, 资金抢筹美债, 此时VIX向上突破看多美债。
         常态下因子休眠为 0.0, 满足 5%-15% Trigger Rate 铁律。
    数据: vixcls, gvzcls
    触发: 极值 (126日 Z-Score/绝对阈值) + 衰竭/突破 (diff < 0 且低于 3日均线)
    输出: +1.0 (看多TLT) 或 -1.0 (看空TLT) 脉冲信号
    """

    def __init__(self):
        self.name = 'vol_crowding_reversal_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值休眠信号
        signal = pd.Series(0.0, index=data.index)

        # 检查依赖数据是否存在
        required_cols = ['vixcls', 'gvzcls']
        for col in required_cols:
            if col not in data.columns:
                return signal

        # 预处理数据 (前向填充避免节假日空值)
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()

        # 计算 126 日滚动 Z-Score (约半年窗口, 适应波动率中枢的平移)
        vix_z = (vix - vix.rolling(window=126).mean()) / vix.rolling(window=126).std()
        gvz_z = (gvz - gvz.rolling(window=126).mean()) / gvz.rolling(window=126).std()
        gvz_z = gvz_z.fillna(0.0)  # 防止早期 GVZ 缺失影响逻辑

        # 二阶导数铁律: 动量与均线衰竭/突破确认
        vix_3d_ma = vix.rolling(window=3).mean()
        
        # 衰竭: 绝对值下降 且 跌破3日均线
        vix_falling = (vix.diff(1) < 0) & (vix < vix_3d_ma)
        
        # 突破: 绝对值上升 且 突破3日均线
        vix_rising = (vix.diff(1) > 0) & (vix > vix_3d_ma)

        # 政权 1: 极度恐慌衰竭 -> 流动性恢复 -> 看多美债 (+1.0)
        # 极高波动率或跨资产恐慌(黄金波动率极端)极值 + 开始衰竭
        sys_panic = (vix_z > 2.0) | (vix.shift(1) >= 32.0) | (gvz_z > 2.5)
        long_regime1 = sys_panic & vix_falling

        # 政权 2: 普通避险衰竭 -> 资金重返风险资产 -> 看空美债 (-1.0)
        # 普通高波动率回调 (排除极度恐慌) + 开始衰竭
        normal_panic = (vix_z > 1.0) & ~sys_panic
        short_regime2 = normal_panic & vix_falling

        # 政权 3: 极度自满破裂 -> 避险情绪骤起 -> 看多美债 (+1.0)
        # 极低波动率 (极度乐观) + 突发向上突破
        complacency = (vix_z < -1.5) & (vix.shift(1) < 16.0)
        long_regime3 = complacency & vix_rising

        # 信号合成
        signal[long_regime1 | long_regime3] = 1.0
        signal[short_regime2] = -1.0

        # 处理冷启动期的 NaN (前126天)
        signal[vix_z.isna()] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"