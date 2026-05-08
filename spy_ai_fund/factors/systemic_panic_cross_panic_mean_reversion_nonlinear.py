import numpy as np
import pandas as pd

class SystemicPanicCrossFactor:
    """系统性恐慌交叉脉冲因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合VIX(波动率)与高收益债利差(信用风险)的非线性特征。当两者同步达到局部极值并出现回落时(二阶导数为负), 确认为系统性恐慌衰竭, 输出强烈看多信号(+1.0); 当两者同步缓慢上升且未达极值时(钝刀割肉期), 市场处于阴跌发酵阶段, 输出看空信号(-1.0)。
    数据: [vixcls, bamlh0a0hym2]
    输出: +1.0 表示系统恐慌见顶衰竭(极佳抄底点), -1.0 表示恐慌情绪发酵恶化(趋势看空), 0.0 为常态休眠
    触发条件: 126日Z-Score>1.5且3日动量转负时触发+1.0; 0.5<Z-Score<=1.5且动量为正时触发-1.0。预期 Trigger Rate 在 8% 到 12% 之间。
    """

    def __init__(self, window: int = 126, z_extreme: float = 1.5, z_mild: float = 0.5, momentum_days: int = 3):
        self.name = 'systemic_panic_cross'
        # 经济学含义: 126天为半个交易年，捕捉中短期维度的宏观情绪极值
        self.window = window
        # 极端偏离度，1.5个标准差约为正常分布的单侧前6.7%极值
        self.z_extreme = z_extreme
        # 轻度偏离度，0.5个标准差代表情绪已明显高于均值但尚未崩溃
        self.z_mild = z_mild
        # 动量观察窗口，3天用于确认趋势的短期转折(二阶导特征)
        self.momentum_days = momentum_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 默认信号输出全 0.0 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        req_cols = ['vixcls', 'bamlh0a0hym2']
        if not all(col in data.columns for col in req_cols):
            return signal
            
        # 前向填充缺失值以对齐频率
        vix = data['vixcls'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()
        
        # 计算 126 日滚动 Z-Score 以识别宏观极值状态
        vix_mean = vix.rolling(self.window, min_periods=self.window//2).mean()
        vix_std = vix.rolling(self.window, min_periods=self.window//2).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-6)
        
        hy_mean = hy_spread.rolling(self.window, min_periods=self.window//2).mean()
        hy_std = hy_spread.rolling(self.window, min_periods=self.window//2).std()
        hy_z = (hy_spread - hy_mean) / (hy_std + 1e-6)
        
        # 计算 3 日动量变化 (边际变化与二阶导数铁律)
        vix_diff = vix.diff(self.momentum_days)
        hy_diff = hy_spread.diff(self.momentum_days)
        
        # ----------------------------------------------------------------------
        # 1. 极端恐慌衰竭 (防接飞刀的抄底买点) -> 输出 +1.0
        # 条件: 波动率或信用利差达到极端状态(Z>1.5)，且二者均开始回落(二阶导为负)
        # ----------------------------------------------------------------------
        is_extreme_stress = ((vix_z > self.z_extreme) & (hy_z > 1.0)) | ((hy_z > self.z_extreme) & (vix_z > 1.0))
        is_exhausted = (vix_diff < 0) & (hy_diff < 0)
        buy_cond = is_extreme_stress & is_exhausted
        
        # ----------------------------------------------------------------------
        # 2. 恐慌发酵/钝刀割肉 (顺势做空/趋势恶化) -> 输出 -1.0
        # 条件: 两者都在上升(动量>0)，且处于中等偏高偏离水平(0.5 < Z <= 1.5)
        # ----------------------------------------------------------------------
        is_mild_stress = (vix_z > self.z_mild) & (vix_z <= self.z_extreme) & (hy_z > self.z_mild) & (hy_z <= self.z_extreme)
        is_worsening = (vix_diff > 0) & (hy_diff > 0)
        sell_cond = is_mild_stress & is_worsening
        
        # 生成脉冲信号
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_extreme={self.z_extreme}, z_mild={self.z_mild}, momentum_days={self.momentum_days})"