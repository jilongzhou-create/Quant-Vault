import numpy as np
import pandas as pd

class VolatilityCrowdingReversalOptionsFactor:
    """波动率极值与拥挤反转 (volatility/options)

    逻辑: 捕捉跨资产波动率的极端状态与反转瞬间。当 VIX 与黄金波动率同步狂飙后首次衰竭，代表流动性去杠杆引发的无差别抛售结束，美债回归避险本质，脉冲看多；当波动率极度低迷滋生大量杠杆头寸，且突然大幅飙升时，代表平静期打破去杠杆开启，脉冲看空美债。
    数据: vixcls (CBOE VIX 波动率), gvzcls (CBOE 黄金隐含波动率)
    触发: 多头 -> VIX 252日 Z-Score > 2.0 且 GVZ Z-Score > 1.0，且 VIX 开始回落(diff < 0 且低于3日均值)；空头 -> VIX Z-Score < -1.0 极度拥挤，且3日内飙升超3点(波动率觉醒)。
    输出: +1.0 看多美债(TLT)，-1.0 看空美债，其余非触发日严格为 0.0。
    """

    def __init__(self):
        self.name = 'volatility_crowding_reversal_options'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 Series，满足零值休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必要字段
        required_cols = ['vixcls', 'gvzcls']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 提取并前向填充数据，防止空值断层
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        if vix.isna().all() or gvz.isna().all():
            return signal
            
        # 计算 252 日滚动 Z-Score (设置最小期数为60，保证初始统计有效性)
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std
        
        gvz_mean = gvz.rolling(window=252, min_periods=60).mean()
        gvz_std = gvz.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        gvz_z = (gvz - gvz_mean) / gvz_std
        
        # 计算边际变化与二阶导数衰竭条件
        vix_3d_mean = vix.rolling(window=3).mean()
        vix_diff_1d = vix.diff(1)
        vix_diff_3d = vix.diff(3)
        
        # 多头脉冲触发逻辑：跨资产恐慌极值 + 衰竭
        # 避免接飞刀：绝对要求 vix_diff_1d < 0 以及 vix < vix_3d_mean
        long_cond = (
            (vix_z > 2.0) & 
            (gvz_z > 1.0) & 
            (vix < vix_3d_mean) & 
            (vix_diff_1d < 0)
        )
        
        # 空头脉冲触发逻辑：极度低波拥挤 + 觉醒
        # 边际变化瞬间：要求 3日内飙升超 3点 且 当天向上
        short_cond = (
            (vix_z < -1.0) & 
            (vix > vix_3d_mean) & 
            (vix_diff_3d > 3.0) & 
            (vix_diff_1d > 0)
        )
        
        # 仅在触发瞬间赋值，严格遵守狙击手脉冲铁律
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        # 处理可能存在的 NaN，确保输出干净
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"