import numpy as np
import pandas as pd

class CreditVixPanicReversionFactor:
    """信用与恐慌极值均值回归因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合股市恐慌(VIX)与信用市场压力(高收益债信用利差)，当系统性恐慌达到极值（Z-Score > 1.0）且双双发生日内衰竭回落时，发出强看多抄底脉冲；而在恐慌初期缓慢上升、钝刀割肉阶段，发出看空脉冲。
    数据: vixcls (VIX指数), bamlh0a0hym2 (美银美林美国高收益债期权调整利差)
    输出: +1.0 表示极端恐慌见顶衰竭（抄底买入），-1.0 表示轻度恐慌持续蔓延（趋势恶化看空），0.0 为无信号
    触发条件: 极值(Z-Score>1.0)+动量转负触发看多，常态震荡+连续多日双升触发看空，预期Trigger Rate约在8%-12%
    """

    def __init__(self, lookback_window: int = 252, extreme_z_score: float = 1.0):
        self.name = 'credit_vix_panic_reversion'
        self.lookback_window = lookback_window
        self.extreme_z_score = extreme_z_score

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()
        
        # 计算基于252个交易日(约一年)的动态Z-Score
        vix_mean = vix.rolling(window=self.lookback_window, min_periods=min(60, self.lookback_window)).mean()
        vix_std = vix.rolling(window=self.lookback_window, min_periods=min(60, self.lookback_window)).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        hy_mean = hy_spread.rolling(window=self.lookback_window, min_periods=min(60, self.lookback_window)).mean()
        hy_std = hy_spread.rolling(window=self.lookback_window, min_periods=min(60, self.lookback_window)).std()
        hy_z = (hy_spread - hy_mean) / hy_std.replace(0, np.nan)
        
        # -----------------------------------------------------
        # 极度恐慌 + 衰竭 = 抄底买入脉冲 (+1.0)
        # -----------------------------------------------------
        # 条件1: 任意一个市场处于极端恐慌状态 (突破Z-Score极值)
        extreme_panic = (vix_z > self.extreme_z_score) | (hy_z > self.extreme_z_score)
        
        # 条件2: 恐慌动量见顶回落 (二阶导数铁律: 绝对禁止直接买入极值，必须等衰竭)
        # 股市恐慌衰竭：今日VIX下降，且跌破过去3日均值
        vix_exhaustion = (vix.diff(1) < 0) & (vix < vix.rolling(3).mean())
        # 信用恐慌衰竭：利差日内收窄
        hy_exhaustion = hy_spread.diff(1) < 0
        
        buy_cond = extreme_panic & vix_exhaustion & hy_exhaustion
        
        # -----------------------------------------------------
        # 轻微恐慌 + 连续升温 = 趋势恶化看空脉冲 (-1.0)
        # -----------------------------------------------------
        # 条件1: 处于轻微恐慌或刚开始恐慌的阶段 (Z-Score > 0 但未到极值)
        creeping_fear_level = (vix_z > 0.0) & (vix_z <= self.extreme_z_score)
        
        # 条件2: 阴跌模式，恐慌指标稳步双升
        # VIX今日上升，且较3日前也在上升，同时信用利差也在走阔
        creeping_fear_trend = (vix.diff(1) > 0) & (vix.diff(3) > 0) & (hy_spread.diff(1) > 0)
        
        sell_cond = creeping_fear_level & creeping_fear_trend
        
        # 合并信号
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback_window={self.lookback_window}, extreme_z_score={self.extreme_z_score})"