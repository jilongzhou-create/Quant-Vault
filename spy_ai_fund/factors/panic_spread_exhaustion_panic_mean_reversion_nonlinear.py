import numpy as np
import pandas as pd

class PanicSpreadExhaustionFactor:
    """恐慌信用利差极值衰竭因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合VIX(股票恐慌)与高收益债信用利差(信用市场恐慌)。极度恐慌且两者共振回落时为极佳抄底点；两者轻度走阔且无衰竭迹象时为钝刀割肉的看空点。
    数据: vixcls (VIX指数), bamlh0a0hym2 (高收益债OAS利差)
    输出: +1.0 (恐慌见顶衰竭，强烈看多), -1.0 (恐慌蔓延期，看空), 0.0 (常态)
    触发条件: 抄底要求季度Z-Score显著偏高且短线拐头向下；看空要求Z-Score中等偏高且短线连续走阔。预期 Trigger Rate 5%-15%。
    """

    def __init__(self, window=63, z_vix_high=1.5, z_spread_high=1.0):
        # 63个交易日对应一个季度的宏观记忆窗口
        self.name = 'panic_spread_exhaustion_pulse'
        self.window = window
        self.z_vix_high = z_vix_high
        self.z_spread_high = z_spread_high

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必需的字段
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return signal
            
        # 向前填充缺失值以防止计算中断
        vix = data['vixcls'].ffill()
        spread = data['bamlh0a0hym2'].ffill()
        
        # 1. 计算季度级别的 Z-Score (识别偏离均值的程度)
        vix_mean = vix.rolling(self.window).mean()
        vix_std = vix.rolling(self.window).std()
        vix_z = (vix - vix_mean) / vix_std
        
        spread_mean = spread.rolling(self.window).mean()
        spread_std = spread.rolling(self.window).std()
        spread_z = (spread - spread_mean) / spread_std
        
        # 2. 计算边际变化与极短期均线 (识别二阶导数/动量的改变)
        vix_diff1 = vix.diff(1)
        vix_diff3 = vix.diff(3)
        vix_ma3 = vix.rolling(3).mean()
        
        spread_diff1 = spread.diff(1)
        spread_diff3 = spread.diff(3)
        
        # 3. 抄底信号 (+1.0): 极值 + 衰竭
        # 物理法则: 股票波动率与信用市场同时承压(Z-Score极高)，但今天动量开始反转回落(差分为负)，流动性冲击边际解除
        long_cond = (
            (vix_z > self.z_vix_high) & 
            (spread_z > self.z_spread_high) & 
            (vix_diff1 < 0) & 
            (vix < vix_ma3) & 
            (spread_diff1 <= 0)
        )
        
        # 4. 看空信号 (-1.0): 趋势恶化期
        # 物理法则: 钝刀割肉阶段，VIX与信用利差在稳步上升(连续3日和1日变化为正)，刚偏离均值但未达极值，属于主跌浪进行中
        short_cond = (
            (vix_z > 0.5) & (vix_z <= self.z_vix_high) &
            (spread_z > 0.5) &
            (vix_diff3 > 0) &
            (spread_diff3 > 0) &
            (vix_diff1 > 0) &
            (spread_diff1 > 0)
        )
        
        # 填入脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        # 处理可能因窗口导致的 NaN
        signal = signal.fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_vix_high={self.z_vix_high}, z_spread_high={self.z_spread_high})"