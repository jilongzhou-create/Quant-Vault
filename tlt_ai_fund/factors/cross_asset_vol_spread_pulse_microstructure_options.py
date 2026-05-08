import numpy as np
import pandas as pd

class CrossAssetVolSpreadPulseFactor:
    """跨资产波动率极值衰竭因子 (microstructure/options)

    逻辑: 捕捉股票隐含波动率(VIX)与黄金隐含波动率(GVZ)的微观结构分化。VIX代表风险资产恐慌，GVZ代表避险/抗通胀资产恐慌。差值(VIX-GVZ)飙升代表纯粹的流动性危机与去杠杆，长债在此期间常遭抛售错杀。当差值达到极端高位且开始回落(二阶导衰竭)时，意味着流动性冲击见顶，资金重新配置美债，输出做多脉冲；反之，当差值极低(通常是恶性通胀/滞胀恐慌，黄金IV畸高)且开始反弹时，输出看空脉冲。因子严格遵循零值休眠，仅在拐点瞬间触发。
    数据: vixcls, gvzcls
    触发: (VIX-GVZ) 126日 Z-Score > 2.5 且 当前值 < 3日均值 → +1.0；Z-Score < -2.5 且 当前值 > 3日均值 → -1.0。
    输出: 脉冲型信号，触发时为 +1.0 或 -1.0，其余时间严格为 0.0。
    """

    def __init__(self, window=126, z_threshold=2.5, smooth=3):
        self.name = 'cross_asset_vol_spread_pulse'
        self.window = window
        self.z_threshold = z_threshold
        self.smooth = smooth

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态输出严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产隐含波动率差值
        spread = vix - gvz
        
        # 计算 Z-Score，使用半年(126个交易日)窗口捕捉宏观维度的异常偏离
        spread_mean = spread.rolling(window=self.window, min_periods=self.window // 2).mean()
        spread_std = spread.rolling(window=self.window, min_periods=self.window // 2).std()
        
        spread_std = spread_std.replace(0, np.nan)
        z_score = (spread - spread_mean) / spread_std
        
        # 计算短期均值用于二阶导数衰竭判断
        short_ma = spread.rolling(window=self.smooth, min_periods=1).mean()
        
        # 看多脉冲逻辑 (铁律2: 二阶导数 + 极值衰竭)
        # 条件1: 流动性恐慌极值 (Z-Score > 2.5)
        extreme_high = z_score > self.z_threshold
        # 条件2: 恐慌开始回落 (边际变化，当前值 < 3日均值)
        exhaustion_high = spread < short_ma
        
        buy_cond = extreme_high & exhaustion_high
        
        # 看空脉冲逻辑 (对称的滞胀恐慌极值衰竭)
        # 条件3: 黄金避险/滞胀恐慌极值 (Z-Score < -2.5)
        extreme_low = z_score < -self.z_threshold
        # 条件4: 恐慌开始见底反弹
        exhaustion_low = spread > short_ma
        
        sell_cond = extreme_low & exhaustion_low
        
        # 赋值脉冲信号
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold}, smooth={self.smooth})"