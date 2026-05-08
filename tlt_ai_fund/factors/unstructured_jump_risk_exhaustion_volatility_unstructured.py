import numpy as np
import pandas as pd

class UnstructuredJumpRiskExhaustionFactor:
    """因子名称 (volatility/unstructured)

    逻辑: 基于非结构化新闻文本提取的宏观跳跃风险(Jump Risk)构建。当跨资产波动率和跳跃风险因黑天鹅危机狂飙至极端高位时，市场流动性枯竭往往导致美债被无差别抛售(Cash is King)；当跳跃风险和VIX同步开始回落(二阶导数衰竭确认)时，标志着流动性危机解除且情绪触顶，避险资金将疯狂涌入美债，触发纯粹的看多脉冲。
    数据: jlnum1m (1个月跳跃风险指数), vixcls (VIX跨资产确认)
    触发: 多头：jlnum1m 252日 Z-Score > 2.5 (极度恐慌) + jlnum1m.diff() < 0 (跳跃衰竭) + vixcls.diff() < 0 (宏观波动同步衰竭)。空头：Z-Score < -2.0 (极度自满) + 二者同步破位上行。
    输出: 脉冲信号 [-1.0, 1.0]，常态完全零值休眠。
    """

    def __init__(self, window=252, z_long=2.5, z_short=-2.0):
        self.name = 'unstructured_jump_risk_exhaustion'
        self.window = window
        self.z_long = z_long
        self.z_short = z_short

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号必须为 0.0 (遵守零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查数据完备性
        if 'jlnum1m' not in data.columns or 'vixcls' not in data.columns:
            return signal
            
        # 前向填充获取有效观察序列
        jr = data['jlnum1m'].ffill()
        vix = data['vixcls'].ffill()
        
        # 计算跳跃风险指数的动态滚动 Z-Score (不使用未来数据)
        jr_mean = jr.rolling(window=self.window, min_periods=self.window//2).mean()
        jr_std = jr.rolling(window=self.window, min_periods=self.window//2).std()
        
        # 防止除 0 问题
        jr_std = jr_std.replace(0.0, np.nan)
        jr_zscore = (jr - jr_mean) / jr_std
        
        # 遵守二阶导数与边际变化铁律：捕捉预期反转瞬间
        jr_diff = jr.diff()
        vix_diff = vix.diff()
        
        # 做多脉冲: 极度恐慌高点 + 同步衰竭 (Anti-Catch-Falling-Knife)
        long_cond = (
            (jr_zscore > self.z_long) & 
            (jr_diff < 0) & 
            (vix_diff < 0)
        )
        
        # 做空脉冲: 极度自满低谷 + 突发反弹打破平静
        short_cond = (
            (jr_zscore < self.z_short) & 
            (jr_diff > 0) & 
            (vix_diff > 0)
        )
        
        # 仅在触发日赋予脉冲极值，否则维持休眠0值
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_long={self.z_long}, z_short={self.z_short})"