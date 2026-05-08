import numpy as np
import pandas as pd

class CrossAssetVolatilitySpreadFactor:
    """跨资产波动率极值衰竭因子 (microstructure/options)

    逻辑: 利用 VIX(美股期权隐含波动率) 与 GVZCLS(黄金期权隐含波动率) 的差值衡量跨资产流动性恐慌微观结构。当该差值极度飙升时代表遭遇无差别抛售的极端现金荒；而当极值开始衰竭回落时，标志着流动性危机消退或央行已兜底，是胜率极高的美债(TLT)反弹介入点。
    数据: vixcls, gvzcls
    触发: 股金波动率差值的 126日 Z-Score > 2.0 (极端恐慌) 且 当日差值跌破过去3日均值 (二阶导数衰竭)
    输出: +1.0 (恐慌见顶消退的脉冲日，看多美债)，其余时间 0.0
    """

    def __init__(self, zscore_window=126, zscore_threshold=2.0, exhaust_window=3):
        self.name = 'cross_asset_vol_spread_exhaustion'
        self.zscore_window = zscore_window
        self.zscore_threshold = zscore_threshold
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 波动率微观结构: 股金隐含波动率差 (衡量流动性挤兑程度)
        vol_spread = vix - gvz
        
        # 计算差值的动态 Z-Score (边际极值)
        spread_mean = vol_spread.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).mean()
        spread_std = vol_spread.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).std()
        z_score = (vol_spread - spread_mean) / spread_std.replace(0, np.nan)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)，必须等待衰竭
        # 衰竭条件: 今日波动率差值 < 过去3日均值
        exhaustion = vol_spread < vol_spread.rolling(window=self.exhaust_window).mean()
        
        # 触发脉冲信号: 极值 + 衰竭
        pulse_trigger = (z_score > self.zscore_threshold) & exhaustion
        
        signal.loc[pulse_trigger] = 1.0
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, zscore_threshold={self.zscore_threshold}, exhaust_window={self.exhaust_window})"