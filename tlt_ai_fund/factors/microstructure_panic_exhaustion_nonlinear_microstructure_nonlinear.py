import numpy as np
import pandas as pd

class MicrostructurePanicExhaustionNonlinearFactor:
    """微观结构/非线性: 合成流动性恐慌极值与衰竭反转因子

    逻辑: 将高频情绪指标(VIX)与低频金融压力指标(NFCI)进行非线性特征交叉, 合成高维微观流动性恐慌指数。仅当合成指数处于极端高位(Z>2.5, 表明市场发生无差别抛售飞刀)且出现确定性边际回落时(二阶导数为负), 捕捉恐慌衰竭瞬间的抄底长债信号。反之, 在极端贪婪被打破瞬间看空。
    数据: vixcls, nfci
    触发: (VIX_Z + NFCI_Z) > 2.5 且合成指标 diff < 0 -> +1.0; 极低位 < -2.0 且抬头 -> -1.0
    输出: 严格脉冲型, +1.0 表示恐慌衰竭看多 TLT, -1.0 表示平静期突变看空 TLT, 正常为 0.0 狙击手休眠
    """

    def __init__(self):
        self.name = 'microstructure_panic_exhaustion_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠, 初始全 0.0, 只有脉冲日触发
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必需字段是否存在
        required_cols = ['vixcls', 'nfci']
        if not all(col in data.columns for col in required_cols):
            return signal

        # 1. 提取并前向填充数据，对齐周频/日频的时间差
        vix = data['vixcls'].ffill()
        nfci = data['nfci'].ffill()

        # 2. 计算半年滚动(126交易日)的动态 Z-Score 水位
        # 使用 min_periods=21 保证起步阶段的鲁棒性, 避免魔法数字
        vix_mean = vix.rolling(window=126, min_periods=21).mean()
        vix_std = vix.rolling(window=126, min_periods=21).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std

        nfci_mean = nfci.rolling(window=126, min_periods=21).mean()
        nfci_std = nfci.rolling(window=126, min_periods=21).std().replace(0, np.nan)
        nfci_z = (nfci - nfci_mean) / nfci_std

        # 3. 非线性交叉：合成高维微观流动性恐慌特征
        # NFCI 提供宏观流动性压力的底色, VIX 提供日频情绪的敏锐波动
        panic_composite = vix_z + nfci_z

        # 4. 二阶导数条件：衰竭与抬头判定
        # 严格遵守铁律2：禁止接飞刀, 必须包含边际回落条件 (diff < 0 且 < 3日均值)
        panic_exhaustion = (
            (panic_composite.diff() < 0) & 
            (panic_composite < panic_composite.rolling(window=3).mean())
        )
        
        panic_worsening = (
            (panic_composite.diff() > 0) & 
            (panic_composite > panic_composite.rolling(window=3).mean())
        )

        # 5. 边际变化脉冲触发赋值 (铁律3: 只在变化发生瞬间赋值)
        # 买入：合成恐慌极值 > 2.5 且 见顶衰竭
        buy_cond = (panic_composite > 2.5) & panic_exhaustion
        
        # 卖出：极端贪婪(无风险偏好) < -2.0 且 开始恶化抬头
        sell_cond = (panic_composite < -2.0) & panic_worsening

        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"