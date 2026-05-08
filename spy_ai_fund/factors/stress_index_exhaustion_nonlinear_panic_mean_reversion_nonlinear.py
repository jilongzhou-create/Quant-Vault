import numpy as np
import pandas as pd

class StressIndexExhaustionNonlinearFactor:
    """Stress Index Exhaustion (panic_mean_reversion/nonlinear)

    逻辑: 综合股市波动率(VIX)与高收益债利差构建宏观压力指数。极端压力且开始回落时做多(抄底)，中度压力连续恶化时做空(趋势恶化)。
    数据: vixcls, bamlh0a0hym2
    输出: +1.0 强烈看多(恐慌衰竭); -1.0 看空(恐慌攀升); 0.0 正常状态
    触发条件: 压力Z-Score>1.2且两指标日变动转负触发多头脉冲; 压力Z处于0~1.0区间且VIX连升3天触发空头脉冲。预期Trigger Rate控制在 6%-10%
    """

    def __init__(self):
        self.name = 'stress_index_exhaustion_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需数据列是否存在
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        spread = data['bamlh0a0hym2'].ffill()
        
        # 计算 252 日 (约1年) 宏观维度的 Z-Score，设定 63 日 (一季度) 作为最小启动窗口
        vix_mean = vix.rolling(window=252, min_periods=63).mean()
        vix_std = vix.rolling(window=252, min_periods=63).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)
        
        spread_mean = spread.rolling(window=252, min_periods=63).mean()
        spread_std = spread.rolling(window=252, min_periods=63).std()
        spread_z = (spread - spread_mean) / (spread_std + 1e-8)
        
        # 交叉构建宏观压力指数: VIX 代表股票直接情绪, 信用利差代表流动性底层基础
        stress_index = 0.6 * vix_z + 0.4 * spread_z
        
        # 计算二阶导数 (边际变化，遵守二阶导数防接飞刀铁律)
        vix_diff = vix.diff()
        spread_diff = spread.diff()
        
        # 多头脉冲：极端恐慌(组合压力 Z-Score > 1.2) + 恐慌见顶衰竭(VIX和利差均停止恶化转好)
        long_cond = (
            (stress_index > 1.2) & 
            (vix_diff < 0) & 
            (spread_diff <= 0)
        )
        
        # 空头脉冲：温水煮青蛙 (压力处于 0.0 ~ 1.0 的均值上方爬升期)，且出现钝刀割肉式的连升
        short_cond = (
            (vix_z > 0.0) & (vix_z < 1.0) &
            (vix_diff > 0) &
            (vix.shift(1).diff() > 0) &
            (vix.shift(2).diff() > 0) &
            (spread_diff > 0)
        )
        
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"