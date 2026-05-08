import numpy as np
import pandas as pd

class CrossAssetVolPremiumFactor:
    """跨资产波动率溢价因子 (microstructure/options)

    逻辑: 衡量股票市场恐慌(VIX)与黄金避险恐慌(GVZ)的波动率价差。当VIX相对GVZ出现极端溢价且开始回落时，标志着纯粹的金融流动性危机见顶，避险资金将回流长债（脉冲看多）；当价差极端异常（通胀恐慌导致GVZ高企而VIX自满）被打破反弹时，往往标志着紧缩预期引发股债双杀的开端（脉冲看空）。
    数据: vixcls, gvzcls
    触发: 波动率价差的252日 Z-Score > 2.0 且低于3日均值（恐慌衰竭看多）；Z-Score < -2.0 且高于3日均值（自满惊醒看空）
    输出: +1.0 表示流动性危机消退抄底美债，-1.0 表示自满被打破逃顶美债，常态 0.0
    """

    def __init__(self):
        self.name = 'cross_asset_vol_premium_microstructure_options'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化非触发日信号为 0.0，严格遵守零值休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 处理数据缺失的情况
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            signal.name = self.name
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产隐含波动率价差
        vol_spread = vix - gvz
        
        # 计算 252日 (一年交易日) 滚动 Z-Score 判定极值水位
        roll_mean = vol_spread.rolling(window=252, min_periods=126).mean()
        roll_std = vol_spread.rolling(window=252, min_periods=126).std()
        zscore = (vol_spread - roll_mean) / roll_std
        
        # 3日均值作为微观结构的极短期动量/衰竭判定线
        vol_spread_3d_ma = vol_spread.rolling(window=3).mean()
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 条件1: 跨资产恐慌极值 (zscore > 2.0)
        # 条件2: 恐慌开始衰竭回落 (vol_spread < 3日均值)
        long_cond = (zscore > 2.0) & (vol_spread < vol_spread_3d_ma)
        
        # 反向条件: 自满情绪极值 (zscore < -2.0) + 波动率开始抬头 (vol_spread > 3日均值)
        short_cond = (zscore < -2.0) & (vol_spread > vol_spread_3d_ma)
        
        # 狙击手脉冲赋值
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"