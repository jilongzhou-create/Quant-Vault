import numpy as np
import pandas as pd

class GoldVolPanicExhaustionFactor:
    """黄金期权恐慌极值衰竭因子 (microstructure/options)

    逻辑: 黄金ETF隐含波动率(GVZ)反映市场对终极避险资产期权溢价的定价。当全球面临极端的滞胀恐慌或系统性流动性挤兑时，GVZ会极端飙升，此时美债常因无差别抛售被错杀。当GVZ触及极值(Z-score>2.0)且开始衰竭回落时，表明流动性危机见顶，避险资金大举回流美债，触发做多脉冲；当波动率极度低迷(Z-score<-1.5)且开始抬升时，反映尾部风险重估，资金重新转向实物避险而抛售美债，触发做空脉冲。常态信号为0。
    数据: gvzcls (CBOE黄金ETF隐含波动率指数)
    触发: 多头脉冲: 252日 Z-Score > 2.0 且 当日值 < 过去3日均值 (恐慌衰竭); 空头脉冲: 252日 Z-Score < -1.5 且 当日值 > 过去3日均值 (尾部风险重估)
    输出: +1.0(看多美债) / -1.0(看空美债) / 0.0(常态休眠)
    """

    def __init__(self, zscore_window: int = 252, extreme_high: float = 2.0, extreme_low: float = -1.5, smooth_window: int = 3):
        self.name = 'gold_vol_panic_exhaustion'
        self.zscore_window = zscore_window
        self.extreme_high = extreme_high
        self.extreme_low = extreme_low
        self.smooth_window = smooth_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'gvzcls' not in data.columns:
            return signal
            
        # 前向填充缺失值以保持时间序列连贯性
        gvz = data['gvzcls'].ffill()
        
        # 计算 252 个交易日的滚动 Z-Score 提取极限情绪
        roll_mean = gvz.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).mean()
        roll_std = gvz.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).std()
        
        # 避免除以零导致无穷大
        roll_std = roll_std.replace(0, np.nan)
        zscore = (gvz - roll_mean) / roll_std
        
        # 计算微观边际衰竭所需的短期均值 (二阶导数锚点)
        short_mean = gvz.rolling(window=self.smooth_window, min_periods=2).mean()
        
        # 核心铁律 2: 二阶导数 (极值 + 衰竭)
        # 条件1 (做多): 流动性恐慌极值 (Z > 2.0) 且 恐慌边际减弱 (低于3日均值)
        long_cond = (zscore > self.extreme_high) & (gvz < short_mean)
        
        # 条件2 (做空): 波动率极度低估 (Z < -1.5) 且 避险溢价抬头 (高于3日均值)
        short_cond = (zscore < self.extreme_low) & (gvz > short_mean)
        
        # 核心铁律 1: 零值休眠，仅脉冲时刻赋值
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, extreme_high={self.extreme_high}, extreme_low={self.extreme_low})"