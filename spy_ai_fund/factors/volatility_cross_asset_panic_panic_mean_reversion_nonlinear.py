import numpy as np
import pandas as pd

class VolatilityCrossAssetPanicFactor:
    """VIX & GVZ Liquidity Panic Exhaustion (panic_mean_reversion/nonlinear)

    逻辑: 专注波动率跨资产交叉。当美股恐慌(VIX)与避险资产恐慌(GVZ-黄金波动率)同时飙升, 代表市场发生“无差别抛售”的极度流动性危机。遵循二阶导数抄底铁律：当极度恐慌且今日见顶回落时, 产生强烈做多信号(流动性冲击衰竭); 而轻微恐慌且未见顶时(阴跌格局), 产生看空信号。
    数据: vixcls (标普500波动率), gvzcls (黄金波动率)
    输出: +1.0 (极度恐慌衰竭, 绝佳抄底买点), -1.0 (轻度恐慌发酵, 趋势恶化), 常态 0.0
    触发条件: VIX_Z > 1.2 且 GVZ_Z > 0.8 且 VIX.diff(1) < 0 时做多; VIX_Z < 1.0 且 4日缓升 > 2.5 时做空。预期 Trigger Rate 控制在 8% - 15%。
    """

    def __init__(self):
        self.name = 'volatility_cross_asset_panic_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # Check required columns
        required_cols = ['vixcls', 'gvzcls']
        if not all(col in data.columns for col in required_cols):
            return pd.Series(0.0, index=data.index)
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        signal = pd.Series(0.0, index=data.index)
        
        if vix.isna().all() or gvz.isna().all():
            return signal
            
        # Z-Score computation (252-day window, proxy for 1 trading year)
        # 用滚动Z-Score识别当前的极值状态，避免使用硬编码的绝对数值
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std().replace(0, 1e-5)
        vix_z = (vix - vix_mean) / vix_std
        
        gvz_mean = gvz.rolling(window=252, min_periods=60).mean()
        gvz_std = gvz.rolling(window=252, min_periods=60).std().replace(0, 1e-5)
        gvz_z = (gvz - gvz_mean) / gvz_std
        
        # 1. 抄底信号 (Long Pulse): 极度恐慌 + 动量衰竭 (严格的二阶导数过滤)
        # 经济学含义: 股票与黄金波动率同步走高代表流动性挤兑，当VIX今日开始回落，代表挤兑抛压断层，大盘即将报复性反弹
        long_cond = (
            (vix_z > 1.2) & 
            (gvz_z > 0.8) & 
            (vix.diff(1) < 0) & 
            (vix < vix.rolling(window=3).mean())
        )
        
        # 2. 看空信号 (Short Pulse): 轻度恐慌缓慢发酵 (钝刀割肉格局)
        # 经济学含义: VIX处于常态偏上(但不极端)，且近期呈现持续上升趋势，典型的市场资金慢性流出，长牛趋势破位初期
        short_cond = (
            (vix_z > 0.0) & (vix_z < 1.0) & 
            (vix.diff(4) > 2.5) & 
            (vix.diff(1) > 0)
        )
        
        # 赋值脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"