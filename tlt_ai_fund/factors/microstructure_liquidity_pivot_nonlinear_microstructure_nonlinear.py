import numpy as np
import pandas as pd

class MicrostructureLiquidityPivotNonlinearFactor:
    """微观流动性枢纽与恐慌衰竭非线性交叉因子 (microstructure/nonlinear)

    逻辑: 将前端资金压力(NFCI)、市场恐慌极值(VIX)与美债微观结构(T10Y2Y动量)进行高维交叉。
          当融资市场极度紧缩或恐慌情绪达到极值，且同时出现衰竭迹象，并伴随收益率曲线
          因短端流动性预期边际改善而开始变陡(Bull Steepener)时，触发做多美债(TLT)的脉冲信号。
          纯脉冲设计，绝对避免在主跌浪中接飞刀。
    数据: vixcls (波动率), nfci (金融压力/融资微观结构), t10y2y (收益率曲线)
    触发: (指标 252日 Z-Score > 2.5) AND (VIX跌破3日均值 且 NFCI停止恶化) AND (曲线动量陡峭)
    输出: +1.0 表示多重流动性恐慌见顶衰竭，看多美债；-1.0 表示极度自满反转，看空美债。常态为 0.0。
    """

    def __init__(self):
        self.name = 'microstructure_liquidity_pivot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始休眠信号 (铁律1: 零值休眠)
        signal = pd.Series(0.0, index=data.index)

        # 检查所需数据列是否存在
        required_cols = ['vixcls', 'nfci', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal

        # 数据前向填充，处理缺失值或低频阶梯状数据(如NFCI是周频)
        vix = data['vixcls'].ffill()
        nfci = data['nfci'].ffill()
        curve = data['t10y2y'].ffill()

        # ---------------- 1. 核心指标与极值计算 (Z-Score) ----------------
        # 遵循 252 日回溯窗口计算标准化极值
        vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
        nfci_z = (nfci - nfci.rolling(252).mean()) / nfci.rolling(252).std()

        # 收益率曲线的微观动量变化 (铁律3: 边际变化，采用5日变化率捕捉瞬间变陡)
        curve_mom = curve.diff(5)
        curve_mom_z = (curve_mom - curve_mom.rolling(252).mean()) / curve_mom.rolling(252).std()

        # ---------------- 2. 做多美债 (TLT) 逻辑 ----------------
        # 条件1: 微观融资压力、恐慌情绪或曲线动量处于极端高位 (极值预警)
        long_extreme = (vix_z > 2.5) | (nfci_z > 2.5) | (curve_mom_z > 2.5)

        # 条件2: 恐慌与流动性压力开始同步衰竭 (铁律2: 二阶导数反转)
        # VIX 必须跌穿过去3日均值 (高位回落)
        vix_exhaust = vix < vix.rolling(3).mean()
        # NFCI 边际变化不再恶化，使用 <= 0 完美兼容其周更后日度 ffill 的常态(值为0)
        nfci_exhaust = nfci.diff() <= 0.0

        # 条件3: 收益率曲线出现短端驱动的变陡确认 (Bull Steepening)
        bull_steepening = curve_mom > 0.0

        # 非线性特征交叉，必须全部满足
        long_cond = long_extreme & vix_exhaust & nfci_exhaust & bull_steepening

        # ---------------- 3. 做空美债 (TLT) 逻辑 ----------------
        # 条件1: 极度自满预警 (波动率极低或金融条件极度宽松)
        short_extreme = (vix_z < -2.0) | (nfci_z < -2.0) | (curve_mom_z < -2.0)

        # 条件2: 自满情绪打破，二阶导数向上反转
        vix_short_exhaust = vix > vix.rolling(3).mean()
        nfci_short_exhaust = nfci.diff() >= 0.0

        # 条件3: 收益率曲线出现平坦化确认 (Bear Flattening 隐含短端收紧)
        bear_flattening = curve_mom < 0.0

        short_cond = short_extreme & vix_short_exhaust & nfci_short_exhaust & bear_flattening

        # ---------------- 4. 信号输出 ----------------
        # 严格按照脉冲铁律赋值，非触发日绝对保持 0.0
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"