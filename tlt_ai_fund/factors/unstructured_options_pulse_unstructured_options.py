import numpy as np
import pandas as pd

class UnstructuredOptionsPulseFactor:
    """Unstructured Options Pulse (unstructured/options)

    逻辑: 结合非结构化政策不确定性数据(USEPU/FOMC)与期权隐含波动率(VIX/GVZ)的极端微观结构。
          期权波动率或政策不确定性指标触及极端高位(Z-Score>2.5)且开始衰竭时，预示着市场恐慌消退及美联储可能开启“Fed Put”注入流动性，触发做多美债脉冲(+1.0)；
          相反，在极度自满(Z<-2.0)被打破时触发做空脉冲(-1.0)。常态下绝对休眠。
    数据: vixcls, gvzcls, usepuindxd, fomc_sentiment
    触发: 波动率及高频非结构化数据采用(63日Z-Score极值 + 跌破3日均值)；低频阶梯数据采用(5日边际变化Z-Score极值 + 仅触发首日)。
    输出: 脉冲信号 [-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'unstructured_options_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (默认全0，常态不交易)
        signal = pd.Series(0.0, index=data.index)
        
        # 1. Options Volatility (VIX) - 波动率微观结构衰竭
        if 'vixcls' in data.columns:
            vix = data['vixcls'].ffill()
            # 63个交易日约为一个宏观自然季度
            vix_std = vix.rolling(63).std().replace(0, np.nan)
            vix_z = (vix - vix.rolling(63).mean()) / vix_std
            
            # 铁律2: 二阶导数 (极值 + 开始衰竭)
            vix_buy = (vix_z > 2.5) & (vix < vix.rolling(3).mean())
            vix_sell = (vix_z < -2.0) & (vix > vix.rolling(3).mean())
            
            signal[vix_buy] = 1.0
            signal[vix_sell] = -1.0
            
        # 2. Options Volatility (GVZ - 黄金跨资产避险情绪期权)
        if 'gvzcls' in data.columns:
            gvz = data['gvzcls'].ffill()
            gvz_std = gvz.rolling(63).std().replace(0, np.nan)
            gvz_z = (gvz - gvz.rolling(63).mean()) / gvz_std
            
            gvz_buy = (gvz_z > 2.5) & (gvz < gvz.rolling(3).mean())
            gvz_sell = (gvz_z < -2.0) & (gvz > gvz.rolling(3).mean())
            
            signal[(signal == 0.0) & gvz_buy] = 1.0
            signal[(signal == 0.0) & gvz_sell] = -1.0
            
        # 3. Unstructured Policy Uncertainty (USEPU - 经济政策不确定性)
        if 'usepuindxd' in data.columns:
            usepu = data['usepuindxd'].ffill()
            usepu_std = usepu.rolling(63).std().replace(0, np.nan)
            usepu_z = (usepu - usepu.rolling(63).mean()) / usepu_std
            
            usepu_buy = (usepu_z > 2.5) & (usepu < usepu.rolling(3).mean())
            usepu_sell = (usepu_z < -2.0) & (usepu > usepu.rolling(3).mean())
            
            signal[(signal == 0.0) & usepu_buy] = 1.0
            signal[(signal == 0.0) & usepu_sell] = -1.0
            
        # 4. Unstructured FOMC Sentiment Shock (低频阶梯情绪得分)
        if 'fomc_sentiment' in data.columns:
            fomc = data['fomc_sentiment'].ffill()
            
            # 铁律3: 边际变化 (绝对禁止使用阶梯绝对值，改用5日动量变化)
            fomc_diff = fomc.diff(5)
            # 126日约为半年，用于平滑低频阶梯数据的均值/方差
            fomc_std = fomc_diff.rolling(126).std().replace(0, np.nan)
            fomc_z = (fomc_diff - fomc_diff.rolling(126).mean()) / fomc_std
            
            # 鸽派突变(>2.5)与鹰派突变(<-2.5)，并通过 .shift(1) 拦截确保只在爆发首日输出1次脉冲
            fomc_buy = (fomc_z > 2.5) & (fomc_z.shift(1) <= 2.5)
            fomc_sell = (fomc_z < -2.5) & (fomc_z.shift(1) >= -2.5)
            
            signal[(signal == 0.0) & fomc_buy] = 1.0
            signal[(signal == 0.0) & fomc_sell] = -1.0
            
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}()"