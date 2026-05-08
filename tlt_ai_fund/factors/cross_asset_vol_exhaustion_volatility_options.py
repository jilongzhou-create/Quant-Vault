import numpy as np
import pandas as pd

class CrossAssetVolExhaustionFactor:
    """跨资产波动率恐慌衰竭因子 (volatility/options)

    逻辑: 监控股市(VIX)与黄金(GVZCLS)期权隐含波动率的共振极值。极端恐慌期往往伴随流动性危机导致的抛售一切行为(包括避险资产美债)。当跨资产波动率从历史极端高位共振衰竭回落时, 确认流动性冲击解除, 避险/政策救市资金重新推升美债, 触发做多脉冲; 反之, 波动率从极度自满的冰点起跳时触发看空脉冲。
    数据: vixcls, gvzcls
    触发: 多头 -> VIX 252日 Z-Score > 2.5 且 GVZ Z-Score > 1.5, 且两者边际同步回落(VIX低于3日均线且日差分<0); 空头 -> VIX 极度低迷(Z-Score < -2.0)且开始向上反弹。
    输出: 脉冲信号, +1.0 表示恐慌衰竭看多美债, -1.0 表示自满破灭看空美债, 其余非触发日一律 0.0。
    """

    def __init__(
        self, 
        lookback_window: int = 252, 
        short_ma: int = 3, 
        z_long_vix: float = 2.5, 
        z_long_gvz: float = 1.5, 
        z_short_vix: float = -2.0
    ):
        self.name = 'cross_asset_vol_exhaustion'
        self.lookback_window = lookback_window
        self.short_ma = short_ma
        self.z_long_vix = z_long_vix
        self.z_long_gvz = z_long_gvz
        self.z_short_vix = z_short_vix

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖列是否存在
        required_cols = ['vixcls', 'gvzcls']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 数据预处理
        df = data[required_cols].ffill()
        vix = df['vixcls']
        gvz = df['gvzcls']
        
        # 计算 252日滚动 Z-Score (年度基准水位)
        vix_mean = vix.rolling(window=self.lookback_window).mean()
        vix_std = vix.rolling(window=self.lookback_window).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        gvz_mean = gvz.rolling(window=self.lookback_window).mean()
        gvz_std = gvz.rolling(window=self.lookback_window).std()
        gvz_z = (gvz - gvz_mean) / gvz_std.replace(0, np.nan)
        
        # 计算动量变化/衰竭指标 (二阶导数 & 边际变化)
        vix_ma_short = vix.rolling(window=self.short_ma).mean()
        vix_diff = vix.diff()
        gvz_diff = gvz.diff()
        
        # --- 狙击手触发逻辑: 看多脉冲 (+1.0) ---
        # 极值条件: 跨资产恐慌共振飙升
        extreme_panic = (vix_z > self.z_long_vix) & (gvz_z > self.z_long_gvz)
        # 衰竭条件: 波动率边际回落, 流动性挤兑解除
        panic_exhaustion = (vix < vix_ma_short) & (vix_diff < 0) & (gvz_diff <= 0)
        
        long_cond = extreme_panic & panic_exhaustion
        
        # --- 狙击手触发逻辑: 看空脉冲 (-1.0) ---
        # 极值条件: VIX 处于历史极度低谷, 市场绝对自满
        extreme_complacency = (vix_z < self.z_short_vix)
        # 反转条件: 波动率脱离底部起跳, 股债可能双杀
        complacency_break = (vix > vix_ma_short) & (vix_diff > 0)
        
        short_cond = extreme_complacency & complacency_break
        
        # 写入脉冲信号 (默认已是 0.0，严格遵守零值休眠)
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback={self.lookback_window}, short_ma={self.short_ma})"