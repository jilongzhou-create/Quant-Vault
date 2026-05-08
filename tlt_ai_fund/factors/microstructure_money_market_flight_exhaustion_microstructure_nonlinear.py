import numpy as np
import pandas as pd

class MicrostructureMoneyMarketFlightExhaustionFactor:
    """微观结构资金避险反转因子 (microstructure/nonlinear)

    逻辑: 追踪货币市场微观结构(Microstructure)的极端避险溢价(Flight-to-Quality)。当市场发生系统性流动性危机时，机构投资者疯狂囤积3个月期国库券(DTB3)，导致其收益率异常性地大幅低于隔夜联邦基金利率(DFF)。本因子通过将这一微观资金压力指标与跨资产恐慌指标(VIX)进行非线性交叉，并在两者同时达到极值(Z-Score > 2.5)且出现动能边际衰竭时(跌破3日均值且当日回落)，精准输出抄底美债的脉冲信号。
    数据: dff (隔夜无风险利率), dtb3 (3个月国库券收益率), vixcls (VIX波动率)
    触发: 过去5日内 DFF-DTB3 或 VIX 的 Z-Score 突破 2.5 且都处于高位，并在当日双双边际回落 (diff < 0 且 < 3日均值)。
    输出: +1.0 (脉冲型美债看多信号)，常态为 0.0
    """

    def __init__(self):
        self.name = 'microstructure_mm_flight_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        req_cols = ['dff', 'dtb3', 'vixcls']
        for col in req_cols:
            if col not in data.columns:
                return signal

        # 处理数据缺失，保持前值
        df = data[req_cols].ffill()

        # 1. 资金面微观结构压力指标 (Money Market Stress Spread)
        # 利差飙升代表对短期安全资产(T-Bill)的极度渴求导致其收益率塌陷
        mm_spread = df['dff'] - df['dtb3']
        mm_mean = mm_spread.rolling(window=252, min_periods=126).mean()
        mm_std = mm_spread.rolling(window=252, min_periods=126).std() + 1e-6
        mm_zscore = (mm_spread - mm_mean) / mm_std

        # 2. 跨资产恐慌情绪指标 (Cross-Asset Panic Volatility)
        vix = df['vixcls']
        vix_mean = vix.rolling(window=252, min_periods=126).mean()
        vix_std = vix.rolling(window=252, min_periods=126).std() + 1e-6
        vix_zscore = (vix - vix_mean) / vix_std

        # 3. 极端极值条件 (过去5日内至少有一个指标达到超级恐慌阈值 2.5，且两者均处于 1.5 强压水准之上)
        mm_max_5d = mm_zscore.rolling(window=5).max()
        vix_max_5d = vix_zscore.rolling(window=5).max()
        
        is_extreme_panic = (mm_max_5d > 2.5) | (vix_max_5d > 2.5)
        is_broad_panic = (mm_max_5d > 1.5) & (vix_max_5d > 1.5)

        # 4. 二阶导数与边际衰竭条件 (严禁接飞刀)
        # 条件：当天指标变化率必须为负(动量衰退)，且跌破短期3日移动均线
        mm_exhausting = (mm_spread.diff() < 0) & (mm_spread < mm_spread.rolling(window=3).mean())
        vix_exhausting = (vix.diff() < 0) & (vix < vix.rolling(window=3).mean())

        # 5. 非线性特征交叉触发 (同步满足：处于高位危机 + 同步开始衰竭)
        trigger = is_extreme_panic & is_broad_panic & mm_exhausting & vix_exhausting

        # 6. 生成脉冲信号 (Sniper Pulse)
        signal.loc[trigger] = 1.0
        
        signal.name = self.name
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}()"