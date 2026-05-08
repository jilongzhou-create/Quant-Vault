import numpy as np
import pandas as pd

class MacroPanicSteepeningReversalFactor:
    """宏观恐慌衰竭与陡峭化反转因子 (volatility/nonlinear)

    逻辑: 结合跨资产恐慌情绪(VIX+GVZCLS)和利率曲线(T10Y2Y)。当跨资产波动率极高且开始回落, 意味着系统抛售引发的流动性冲击衰竭; 此时若收益率曲线同步确认陡峭化(降息预期形成), 则是极高胜率的美债避险做多脉冲点。反之, 极度自满情绪逆转且曲线平坦化时做空美债。
    数据: vixcls, gvzcls, t10y2y
    触发: 跨资产波动率综合 Z-Score > 1.5 且开始回落(二阶导), 且 T10Y2Y 边际变陡(升破均线) -> +1.0
    输出: 脉冲型信号 [-1.0, 1.0], 正值看多美债(TLT)
    """

    def __init__(self, window=252, z_threshold=1.5, smooth_window=5):
        self.name = 'macro_panic_steepening_reversal_nonlinear'
        self.window = window
        self.z_threshold = z_threshold
        self.smooth_window = smooth_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值信号 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必要字段是否存在
        req_cols = ['vixcls', 'gvzcls', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # 数据前向填充以处理节假日错位
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 1. 波动率水平特征: 跨资产恐慌 Z-Score 极值评估
        vix_std = vix.rolling(window=self.window).std()
        gvz_std = gvz.rolling(window=self.window).std()
        
        vix_z = (vix - vix.rolling(window=self.window).mean()) / (vix_std + 1e-6)
        gvz_z = (gvz - gvz.rolling(window=self.window).mean()) / (gvz_std + 1e-6)
        
        # 构建跨资产恐慌共振指标
        comp_z = (vix_z + gvz_z) / 2.0
        
        # 2. 衰竭确认特征 (二阶导数铁律: 必须跌破均线且发生负动量才算企稳)
        vix_smooth = vix.rolling(window=self.smooth_window).mean()
        gvz_smooth = gvz.rolling(window=self.smooth_window).mean()
        
        vix_falling = (vix < vix_smooth) & (vix.diff() < 0)
        gvz_falling = (gvz < gvz_smooth) & (gvz.diff() < 0)
        
        vix_rising = (vix > vix_smooth) & (vix.diff() > 0)
        gvz_rising = (gvz > gvz_smooth) & (gvz.diff() > 0)
        
        # 3. 宏观跨域确认: 收益率曲线动量 (边际变化铁律: 捕捉曲线变陡的爆发点)
        t10_smooth = t10y2y.rolling(window=self.smooth_window).mean()
        steepening = (t10y2y > t10_smooth) & (t10y2y.diff() > 0)
        flattening = (t10y2y < t10_smooth) & (t10y2y.diff() < 0)
        
        # 组合高维脉冲触发条件
        # 多头脉冲: 极度恐慌 + 恐慌瓦解 + 短端利率下行致变陡 = 美债暴涨起点
        bull_cond = (comp_z > self.z_threshold) & vix_falling & gvz_falling & steepening
        
        # 空头脉冲: 极度自满 + 风险反转 + 长端利率上行致平坦 = 美债抛售起点
        bear_cond = (comp_z < -self.z_threshold) & vix_rising & gvz_rising & flattening
        
        # 赋值狙击手级脉冲信号
        signal[bull_cond] = 1.0
        signal[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold}, smooth_window={self.smooth_window})"