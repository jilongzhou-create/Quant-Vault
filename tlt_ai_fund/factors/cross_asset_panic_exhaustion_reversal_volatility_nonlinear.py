import numpy as np
import pandas as pd

class CrossAssetPanicExhaustionReversalFactor:
    """跨资产恐慌极值衰竭反转因子 (volatility/nonlinear)

    逻辑: 当股票(VIX)与黄金(GVZCLS)的跨资产波动率同步出现极端狂飙，且金融状况指数(NFCI)恶化时，表明市场陷入流动性拥挤危机。在危机巅峰直接买入美债会死于无差别抛售(接飞刀)，因此必须等波动率指标同时出现边际回落(二阶导数衰竭)、不再恶化时，才确认流动性冲击结束，避险资金安全回流美债，触发狙击级做多脉冲。反之，在极度自满且突然发散飙升时，触发看空脉冲。
    数据: ['vixcls', 'gvzcls', 'nfci']
    触发: 
      - 看多(+1.0): 多维度 Z-Score 极高 (VIX > 2.5 等) 且开始向下衰竭 (diff() < 0 且低于3日均值)。
      - 看空(-1.0): 多维度 Z-Score 极低 (极度自满) 且开始向上突发跳升 (diff() > 0 且高于3日均值)。
    输出: 严格脉冲型信号，[-1.0, 0.0, 1.0]
    """

    def __init__(self):
        self.name = 'cross_asset_panic_exhaustion_reversal'
        self.macro_window = 252  # 经济学含义: 一年的交易日，用于锚定宏观周期的绝对水位

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，默认全 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        required_cols = ['vixcls', 'gvzcls', 'nfci']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 1. 提取并前向填充低频/缺失数据，保证日频对齐
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        nfci = data['nfci'].ffill()
        
        # 2. 计算 252 日滚动 Z-Score (反映宏观周期的极端水位)
        vix_z = (vix - vix.rolling(self.macro_window).mean()) / vix.rolling(self.macro_window).std()
        gvz_z = (gvz - gvz.rolling(self.macro_window).mean()) / gvz.rolling(self.macro_window).std()
        nfci_z = (nfci - nfci.rolling(self.macro_window).mean()) / nfci.rolling(self.macro_window).std()
        
        # 3. 铁律3: 边际变化计算
        vix_diff = vix.diff()
        gvz_diff = gvz.diff()
        
        # 铁律2: 二阶导数使用的短期衰竭基准
        vix_ma3 = vix.rolling(3).mean()
        gvz_ma3 = gvz.rolling(3).mean()
        nfci_ma5 = nfci.rolling(5).mean()  # NFCI 为周频发布，使用5日平滑避免由于阶梯更新导致的假动作
        
        # 4. 狙击手脉冲触发逻辑
        # 条件A: 多重恐慌极值 + 同步边际回落衰竭 = 恐慌消散，避险重构，买入美债(TLT)
        long_extreme = (vix_z > 2.5) & (gvz_z > 1.5) & (nfci_z > 1.0)
        long_exhaustion = (vix_diff < 0) & (vix < vix_ma3) & (gvz_diff < 0) & (gvz < gvz_ma3) & (nfci <= nfci_ma5)
        long_trigger = long_extreme & long_exhaustion
        
        # 条件B: 多重极度自满 + 同步向上跳升爆发 = 紧缩/风险冲击开启，抛售美债(TLT)
        short_extreme = (vix_z < -1.5) & (gvz_z < -1.5) & (nfci_z < -1.0)
        short_exhaustion = (vix_diff > 0) & (vix > vix_ma3) & (gvz_diff > 0) & (gvz > gvz_ma3) & (nfci >= nfci_ma5)
        short_trigger = short_extreme & short_exhaustion
        
        # 5. 赋值极值脉冲信号
        signal.loc[long_trigger] = 1.0
        signal.loc[short_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"