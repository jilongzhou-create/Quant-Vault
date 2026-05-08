import numpy as np
import pandas as pd

class LiquidityPanicExhaustionNonlinearFactor:
    """流动性恐慌衰竭非线性交叉因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合美股波动率(VIX)和避险资产黄金波动率(GVZ)。当两者同时飙升(流动性危机导致的无差别抛售), 且随后动能由正转负时, 标志着极端恐慌衰竭, 是美股极佳的抄底买点; 而当波动率刚脱离均值并持续上升时, 处于轻度恐慌的'主跌浪', 提示做空。
    数据: [vixcls, gvzcls]
    输出: 恐慌极值且衰竭时输出+1.0(看多), 轻微恐慌发酵恶化时输出-1.0(看空), 常态输出0.0
    触发条件: 极端恐慌衰竭(VIX Z>1.5且动能回落)触发+1.0, 恐慌发酵(0.5<Z<1.5且连涨)触发-1.0, 预期 Trigger Rate 约 8%-12%
    """

    def __init__(self, window=252, extreme_z=1.5, mild_z=0.5):
        self.name = 'liquidity_panic_exhaustion_nonlinear'
        self.window = window
        self.extreme_z = extreme_z
        self.mild_z = mild_z

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查是否包含必要字段
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 向前填充缺失值
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算 252 日(约一年)滚动 Z-Score
        vix_mean = vix.rolling(window=self.window, min_periods=self.window//2).mean()
        vix_std = vix.rolling(window=self.window, min_periods=self.window//2).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        gvz_mean = gvz.rolling(window=self.window, min_periods=self.window//2).mean()
        gvz_std = gvz.rolling(window=self.window, min_periods=self.window//2).std()
        gvz_z = (gvz - gvz_mean) / gvz_std.replace(0, np.nan)
        
        # 计算边际变化 (二阶导数铁律)
        # 3日变化率代表极端脉冲后的短期趋势回落
        vix_diff_3d = vix.diff(3)
        gvz_diff_3d = gvz.diff(3)
        
        # 1日连续变化代表轻度恐慌过程中的发酵发散
        vix_diff_1 = vix.diff(1)
        vix_diff_1_prev = vix.shift(1).diff(1)
        
        # 1. 强看多信号 (+1.0): 抄底极端流动性恐慌的衰竭时刻
        # 逻辑: VIX处于历史极高位(Z > 1.5), 且避险资产GVZ也处于高位(Z > 1.0 说明避险同样遭抛售), 
        # 且两者均出现见顶回落(3日动能由正转负)
        long_cond = (
            (vix_z > self.extreme_z) & 
            (gvz_z > 1.0) & 
            (vix_diff_3d < 0) & 
            (gvz_diff_3d < 0)
        )
        
        # 2. 强看空信号 (-1.0): 警惕主跌浪(轻度恐慌发酵)
        # 逻辑: VIX刚刚脱离常态(0.5 < Z <= 1.5), 且连续两天上涨, 
        # GVZ也同步升高(Z > 0.5), 说明恐慌正在恶化发酵, 此时接飞刀必死
        short_cond = (
            (vix_z > self.mild_z) & 
            (vix_z <= self.extreme_z) & 
            (vix_diff_1 > 0) & 
            (vix_diff_1_prev > 0) & 
            (gvz_z > self.mild_z)
        )
        
        # 赋值信号输出
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        # 处理可能产生的 NaN, 确保在安全状态下返回 0.0
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, extreme_z={self.extreme_z}, mild_z={self.mild_z})"