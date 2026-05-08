import numpy as np
import pandas as pd

class FomcVolatilityExhaustionFactor:
    """Fomc Volatility Exhaustion Factor (volatility/unstructured)

    逻辑: 结合非结构化FOMC文本情绪得分的边际突变与跨资产恐慌情绪(VIX/GVZ)的高位衰竭。当美联储情绪发生极端“鹰鸽突变”时，若此时跨资产波动率处于相对高位并同步回落，说明政策不确定性瞬间落地，触发针对美债方向的极值脉冲信号。
    数据: fomc_sentiment, vixcls, gvzcls
    触发: fomc_sentiment.diff() 的 252日 Z-Score 绝对值 > 2.5 AND vixcls Z-Score > 1.0 AND (vixcls 与 gvzcls 均开始衰竭回落)
    输出: +1.0 表示看多美债(政策鸽派突变+恐慌落地), -1.0 表示看空美债(政策鹰派突变+恐慌落地), 其余日常时段输出 0.0
    """

    def __init__(self, fomc_window=3, z_window=252, min_periods=60, fomc_z_thresh=2.5, vix_z_thresh=1.0, exhaust_window=3):
        self.name = 'fomc_volatility_exhaustion_unstructured'
        self.fomc_window = fomc_window
        self.z_window = z_window
        self.min_periods = min_periods
        self.fomc_z_thresh = fomc_z_thresh
        self.vix_z_thresh = vix_z_thresh
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 数据依赖检查
        req_cols = ['fomc_sentiment', 'vixcls', 'gvzcls']
        if not all(col in data.columns for col in req_cols):
            return signal
            
        fomc = data['fomc_sentiment'].ffill()
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 铁律3: 边际变化 (只监控政策情绪预期跳跃的一瞬间，绝不使用绝对水位)
        fomc_diff = fomc.diff(self.fomc_window)
        fomc_diff_mean = fomc_diff.rolling(window=self.z_window, min_periods=self.min_periods).mean()
        fomc_diff_std = fomc_diff.rolling(window=self.z_window, min_periods=self.min_periods).std()
        fomc_z = (fomc_diff - fomc_diff_mean) / (fomc_diff_std + 1e-8)
        
        # 铁律2: 二阶导数之前置条件 - 波动率水位极端且跨资产恐慌共振
        vix_mean = vix.rolling(window=self.z_window, min_periods=self.min_periods).mean()
        vix_std = vix.rolling(window=self.z_window, min_periods=self.min_periods).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)
        
        # 铁律2: 二阶导数之衰竭条件 - 绝不接飞刀，必须等波动率切线向下
        vix_exhaustion = vix < vix.rolling(window=self.exhaust_window).mean()
        gvz_exhaustion = gvz < gvz.rolling(window=self.exhaust_window).mean()
        
        # 铁律1: 狙击手脉冲信号触发逻辑
        # 多头脉冲: 情绪历史级放鸽突变 + 恐慌盘处高位 + 跨资产恐慌开始瓦解
        long_cond = (
            (fomc_z > self.fomc_z_thresh) & 
            (vix_z > self.vix_z_thresh) & 
            vix_exhaustion & 
            gvz_exhaustion
        )
        
        # 空头脉冲: 情绪历史级放鹰突变 + 恐慌盘处高位 + 跨资产恐慌开始瓦解 (利空共识落地)
        short_cond = (
            (fomc_z < -self.fomc_z_thresh) & 
            (vix_z > self.vix_z_thresh) & 
            vix_exhaustion & 
            gvz_exhaustion
        )
        
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"fomc_window={self.fomc_window}, z_window={self.z_window}, "
                f"fomc_z_thresh={self.fomc_z_thresh}, vix_z_thresh={self.vix_z_thresh})")