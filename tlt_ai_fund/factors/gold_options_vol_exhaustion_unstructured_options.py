import numpy as np
import pandas as pd

class OptionsCrossAssetVolRatioFactor:
    """options / cross_asset_vol_ratio

    逻辑: VIX(美股隐含波动率)与GVZ(黄金隐含波动率)的比值反映了"增长恐慌"与"滞胀恐慌"的博弈。当比值极端高且开始衰竭时，代表通缩性流动性冲击(Dash for Cash)缓解，美债作为避险资产将重新获得流动性青睐开启主升浪(看多)；当比值极端低且开始反弹时，代表通胀/信用贬值引发的硬资产恐慌见顶，此时联储将被迫维持结构性紧缩以抗击通胀，美债承压(看空)。
    数据: vixcls, gvzcls
    触发: VIX/GVZ比值的252日Z-Score > 1.5 且 3日内该比值开始回落(边际衰竭) -> 脉冲+1.0；Z-Score < -1.5 且 3日内比值开始回升 -> 脉冲-1.0。
    输出: 脉冲型，满足条件输出+1.0或-1.0，常态为0.0。
    """

    def __init__(self):
        self.name = 'options_cross_asset_vol_ratio'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        # 将0替换为NaN防止除零，尽管隐含波动率极少为0
        gvz = data['gvzcls'].replace(0, np.nan).ffill()
        
        # 计算跨资产波动率比值
        vol_ratio = vix / gvz
        
        # 计算252日(约1个交易年)的Z-Score，包含最低63天(约1个季度)启动
        roll_mean = vol_ratio.rolling(window=252, min_periods=63).mean()
        roll_std = vol_ratio.rolling(window=252, min_periods=63).std()
        roll_std = roll_std.replace(0, np.nan)  # 防除零异常
        
        z_score = (vol_ratio - roll_mean) / roll_std
        
        # 边际变化与极值衰竭条件 (二阶导数铁律)
        # 3日变化量，捕捉极端情绪极值过后的初期拐点
        diff_3d = vol_ratio.diff(3)
        
        # 条件1: 增长恐慌（通缩冲击）衰竭 -> 美债重获避险资金青睐 (Long TLT)
        long_cond = (z_score > 1.5) & (diff_3d < 0)
        
        # 条件2: 滞胀恐慌（通胀冲击）衰竭 -> 硬资产恐慌见顶但联储被迫紧缩，美债熊市中继 (Short TLT)
        short_cond = (z_score < -1.5) & (diff_3d > 0)
        
        # 赋予脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"