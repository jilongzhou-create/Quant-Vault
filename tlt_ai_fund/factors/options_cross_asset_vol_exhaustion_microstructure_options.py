import numpy as np
import pandas as pd

class OptionsCrossAssetVolExhaustionFactor:
    """跨资产期权波动率恐慌衰竭因子 (microstructure/options)

    逻辑: 在流动性危机极值点，股票期权隐含波动率(VIX)和黄金期权隐含波动率(GVZ)会由于"现金为王"的挤兑而同时飙升。当跨资产联合波动率达到极度恐慌水位且动量开始衰竭回落时，标志着无差别抛售结束，随后的宽松预期与避险需求将驱动美债大幅反弹。反之，在极度贪婪(波动率极低)且动量抬头时，预示通胀紧缩或风险平价抛售，看空美债。必须使用脉冲输出以防接飞刀。
    数据: vixcls, gvzcls
    触发: (VIX+GVZ) 252日 Z-Score > 2.5 且开始回落(<3日均值) 则看多；Z-Score < -2.0 且开始抬头(>3日均值) 则看空。
    输出: +1.0 (极度恐慌衰竭，看多美债) / -1.0 (极度贪婪抬头，看空美债)
    """

    def __init__(self):
        self.name = 'options_cross_asset_vol_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查是否包含所需列
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 前向填充缺失值以防止跳空中断
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 合成跨资产期权隐含波动率
        cross_vol = vix + gvz
        
        # 计算 252日(约1年) Z-Score 以捕捉相对极值
        roll_mean = cross_vol.rolling(window=252, min_periods=126).mean()
        roll_std = cross_vol.rolling(window=252, min_periods=126).std()
        
        # 避免除以0或极小值产生无限大
        roll_std = roll_std.replace(0.0, np.nan)
        vol_zscore = (cross_vol - roll_mean) / roll_std
        
        # 计算衰竭/变化条件 (3日移动平均代表短期微观动量)
        vol_3d_ma = cross_vol.rolling(window=3, min_periods=1).mean()
        
        # 条件组 A: 恐慌极值 + 开始衰竭 (看多美债)
        cond_panic_extreme = vol_zscore > 2.5
        cond_panic_exhaustion = cross_vol < vol_3d_ma
        
        # 条件组 B: 贪婪极度低迷 + 开始抬头 (看空美债)
        cond_complacency_extreme = vol_zscore < -2.0
        cond_complacency_reversal = cross_vol > vol_3d_ma
        
        # 信号赋值 (严格遵守脉冲休眠铁律)
        signal.loc[cond_panic_extreme & cond_panic_exhaustion] = 1.0
        signal.loc[cond_complacency_extreme & cond_complacency_reversal] = -1.0
        
        # 填充可能因NaN产生的空值
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"