import numpy as np
import pandas as pd

class CreditVixPanicCrossFactor:
    """信用波动恐慌极值交叉因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合股市恐慌(VIX)和信用市场压力(高收益债利差)的非线性特征。恐慌初期且压力积累时输出看空脉冲；恐慌达到极值且双双出现见顶回落(二阶导数为负)时，捕捉恐慌衰竭点输出强看多抄底脉冲。
    数据: vixcls, bamlh0a0hym2
    输出: +1.0 表示恐慌极值见顶回落，强烈看多；-1.0 表示恐慌中度发酵，趋势恶化看空；0.0 为常态休眠。
    触发条件: 做多条件为VIX与利差的半年Z-Score极高且当日均下降(衰竭)；做空条件为Z-Score在中位但动量大幅上升，预期 Trigger Rate 控制在 8%-12% 之间。
    """

    def __init__(self, zscore_window=126, vix_extreme_z=1.5, hy_extreme_z=1.0):
        self.name = 'credit_vix_panic_cross'
        self.zscore_window = zscore_window
        self.vix_extreme_z = vix_extreme_z
        self.hy_extreme_z = hy_extreme_z

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 默认返回全 0 的休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        # 检查是否包含所需字段
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return signal
            
        # 填充缺失值并获取序列 (不同市场节假日可能导致缺失)
        vix = data['vixcls'].ffill()
        hy = data['bamlh0a0hym2'].ffill()
        
        # 计算 126日 (半年) Z-Score，使用最小周期为一季度(63日)，无未来函数
        vix_mean = vix.rolling(window=self.zscore_window, min_periods=63).mean()
        vix_std = vix.rolling(window=self.zscore_window, min_periods=63).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-6)
        
        hy_mean = hy.rolling(window=self.zscore_window, min_periods=63).mean()
        hy_std = hy.rolling(window=self.zscore_window, min_periods=63).std()
        hy_z = (hy - hy_mean) / (hy_std + 1e-6)
        
        # 计算边际变化与动量 (核心二阶导数识别)
        vix_diff_1 = vix.diff(1)
        hy_diff_1 = hy.diff(1)
        
        # 3日中短期动量变化
        vix_pct_3 = vix.diff(3) / (vix.shift(3) + 1e-6)
        hy_diff_3 = hy.diff(3)
        
        # === 狙击点 1: 恐慌极值 + 衰竭 (强烈看多美股) ===
        # 股市处于过去半年的极度恐慌 (VIX_Z > 1.5) 且今天恐慌回落 (vix_diff_1 < 0)
        # 并且信贷市场同样被恐慌传染 (HY_Z > 1.0) 且今天利差不再走阔 (hy_diff_1 <= 0)
        long_cond = (vix_z > self.vix_extreme_z) & (vix_diff_1 < 0) & \
                    (hy_z > self.hy_extreme_z) & (hy_diff_1 <= 0)
                    
        # === 狙击点 2: 钝刀割肉趋势恶化 (顺势看空美股) ===
        # 恐慌开始上升但尚未接飞刀 (0.5 < VIX_Z <= 1.5，未到极值)
        # 且恐慌正在快速发酵主升浪 (过去3天 VIX 飙升 > 10%)
        # 且信贷基本面同时恶化 (过去3天 高收益债利差走阔 > 10 个基点即0.1)
        short_cond = (vix_z > 0.5) & (vix_z <= self.vix_extreme_z) & \
                     (vix_pct_3 > 0.10) & (hy_diff_3 > 0.10)
                     
        # 生成脉冲信号
        signal.loc[short_cond] = -1.0
        signal.loc[long_cond] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, vix_extreme_z={self.vix_extreme_z}, hy_extreme_z={self.hy_extreme_z})"