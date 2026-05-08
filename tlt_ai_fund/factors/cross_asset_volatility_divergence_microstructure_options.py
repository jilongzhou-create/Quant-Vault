import numpy as np
import pandas as pd

class CrossAssetVolatilityDivergenceFactor:
    """跨资产波动率背离衰竭因子 (microstructure/options)

    逻辑: 股票隐含波动率(VIX)与硬通货隐含波动率(GVZ)的比值反映了宏观冲击的性质。当 VIX 相对 GVZ 极端飙升(典型的流动性/通缩冲击)并在极值处开始回落时，美联储救市预期兑现，流动性危机解除，脉冲做多避险美债(TLT)；当 GVZ 相对 VIX 极端飙升(典型的法币信用/滞胀冲击)并开始反转时，说明市场不再单纯恐慌而是开始计价长期紧缩，杀估值展开，此时脉冲做空美债(TLT)。
    数据: vixcls, gvzcls
    触发: Z-Score(VIX/GVZ) > 2.5 且比值跌破3日均线 -> +1.0 (流动性恐慌衰竭)；Z-Score < -2.5 且比值突破3日均线 -> -1.0 (滞胀恐慌拐点)
    输出: [-1.0, 1.0] 的极短期脉冲信号
    """

    def __init__(self):
        self.name = 'cross_asset_vol_divergence'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 常态下必须是零值休眠，预先填充 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 必须验证所需数据字段是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].replace(0, np.nan).ffill()
        
        # 检查数据是否充分
        if vix.isna().all() or gvz.isna().all():
            return signal
            
        # 计算跨资产波动率比值 (股票风险 vs 实物通胀风险)
        ratio = vix / gvz
        
        # 铁律3: 必须基于边际变化，计算 252日 (一整个自然年) 滚动 Z-Score 衡量极值
        roll_mean = ratio.rolling(window=252, min_periods=126).mean()
        roll_std = ratio.rolling(window=252, min_periods=126).std()
        
        # 避免除零造成的无穷大异常
        roll_std = roll_std.replace(0, np.nan)
        zscore = (ratio - roll_mean) / roll_std
        
        # 铁律2: 二阶导数，绝对禁止直接在极值处接飞刀，必须等动量衰竭 (3日均线作为短期平滑拐点)
        short_term_trend = ratio.rolling(window=3).mean()
        exhaustion_long = ratio < short_term_trend
        exhaustion_short = ratio > short_term_trend
        
        # 狙击手脉冲触发逻辑
        # 条件1: 处于绝对历史高低极值 (Z > 2.5 或 Z < -2.5)
        # 条件2: 刚出现动量衰竭
        long_trigger = (zscore > 2.5) & exhaustion_long
        short_trigger = (zscore < -2.5) & exhaustion_short
        
        signal.loc[long_trigger] = 1.0
        signal.loc[short_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"