import numpy as np
import pandas as pd

class NewsPolicyUncertaintyShockFactor:
    """新闻政策不确定性冲击因子 (Unstructured NLP)

    逻辑: 极端的宏观经济政策不确定性(EPU)飙升往往伴随资产的无差别抛售。当EPU动能飙升至极值且开始衰竭时，标志流动性杀跌结束，避险资金大量切入美债(TLT)，形成看多脉冲。反之，当不确定性断崖式下降且开始反弹时，风险偏好见顶，避险资金撤出长债，形成看空脉冲。
    数据: usepuindxd (美国经济政策不确定性指数，基于新闻文本的NLP情绪数据)
    触发: EPU 10日边际变化量的252日 Z-Score > 2.5 且动量跌破3日均线(衰竭) -> +1.0 (看多)；Z-Score < -2.5 且动量升破3日均线 -> -1.0 (看空)
    输出: [-1.0, 1.0] 的狙击手级脉冲信号
    """

    def __init__(self, z_threshold: float = 2.5, lookback: int = 252, smooth_window: int = 10, mom_window: int = 10):
        self.name = 'news_epu_shock_exhaustion'
        self.z_threshold = z_threshold
        self.lookback = lookback
        self.smooth_window = smooth_window
        self.mom_window = mom_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态信号必须为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            signal.name = self.name
            return signal

        epu = data['usepuindxd'].ffill()
        
        # 为降低非结构化新闻数据的高频日度噪音，先进行基础平滑
        epu_smooth = epu.rolling(window=self.smooth_window).mean()
        
        # 铁律3: 边际变化 Only (严格使用动量变化而非绝对水位)
        epu_mom = epu_smooth.diff(self.mom_window)
        
        # 计算动量的滚动 Z-Score
        roll_mean = epu_mom.rolling(window=self.lookback).mean()
        roll_std = epu_mom.rolling(window=self.lookback).std().replace(0.0, np.nan)
        epu_mom_z = (epu_mom - roll_mean) / roll_std
        
        # 铁律2: 二阶导数防飞刀 (极值 + 开始衰竭)
        # 动量衰竭：当前动量向均值回归 (回落或反弹)
        mom_exhaustion_up = epu_mom < epu_mom.rolling(window=3).mean()
        mom_exhaustion_dn = epu_mom > epu_mom.rolling(window=3).mean()
        
        # 脉冲触发条件组合
        long_cond = (epu_mom_z > self.z_threshold) & mom_exhaustion_up
        short_cond = (epu_mom_z < -self.z_threshold) & mom_exhaustion_dn
        
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_threshold={self.z_threshold}, lookback={self.lookback}, mom_window={self.mom_window})"