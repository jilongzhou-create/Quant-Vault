import numpy as np
import pandas as pd

class MortgageVolShockExhaustionFactor:
    """按揭抵押期权波动率突变衰竭因子 (unstructured/options)

    逻辑: MBS期权隐含波动率(obmmiva30yf)代表了固定收益市场的微观凸性对冲恐慌。当长端利率飙升引发MBS的负凸性抛售(Negative Convexity Hedging)时，MBS隐波会产生极端脉冲式上升。本因子严格遵守三大铁律，通过其5日边际变化的极端飙升(Z-Score>2.5)且出现动量衰竭(单日变动<0且低于3日均值)来捕捉抛售恐慌耗尽的瞬间，输出狙击手级别的看多脉冲(抄底TLT)。反之，捕捉债市极度松懈后的反转看空瞬间。
    数据: obmmiva30yf (ICE BofA 30-Year MBS Option Volatility Estimate)
    触发: 5日边际变化的Z-Score极值(>2.5或<-2.5) + 二阶导数衰竭(波动率反转)
    输出: +1.0 看多美债(抛售恐慌耗尽)，-1.0 看空美债(极度平静被打破)，其余状态为0.0
    """

    def __init__(self, lookback_window: int = 252, diff_window: int = 5):
        self.name = 'mortgage_vol_shock_exhaustion'
        self.lookback_window = lookback_window
        self.diff_window = diff_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (常态无脑为0.0，只在脉冲瞬间输出信号)
        signal = pd.Series(0.0, index=data.index)
        
        # 因子防御: 处理缺少核心数据源的情况
        if 'obmmiva30yf' not in data.columns:
            return signal
            
        # 填充缺失值，避免序列计算断层
        mb_vol = data['obmmiva30yf'].ffill()
        
        # 铁律3: 边际变化 (Only Marginal Change) 绝对禁止对比绝对水位
        vol_diff = mb_vol.diff(self.diff_window)
        
        # 动态计算波动率变动的 Z-Score
        roll_mean = vol_diff.rolling(self.lookback_window, min_periods=self.lookback_window//2).mean()
        roll_std = vol_diff.rolling(self.lookback_window, min_periods=self.lookback_window//2).std()
        
        # 加上微小epsilon避免标准差为0导致的除息错误
        z_score = (vol_diff - roll_mean) / (roll_std + 1e-8)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 飙升衰竭：波动率极速放大到达顶峰后，开始发生边际回落
        surge_exhaustion = (mb_vol < mb_vol.rolling(3).mean()) & (mb_vol.diff(1) < 0)
        
        # 骤降反转：波动率极速压缩后，底部的死水状态开始重新抬头
        plunge_exhaustion = (mb_vol > mb_vol.rolling(3).mean()) & (mb_vol.diff(1) > 0)
        
        # 做多TLT脉冲: MBS波动率变动出现极端飙升(债市暴跌恐慌抛售) AND 恐慌开始边际衰竭
        long_cond = (z_score > 2.5) & surge_exhaustion
        
        # 做空TLT脉冲: MBS波动率变动出现暴跌极值(市场极度贪婪/松懈) AND 波动率底部反弹打破平静
        short_cond = (z_score < -2.5) & plunge_exhaustion
        
        # 赋值狙击手信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback_window={self.lookback_window}, diff_window={self.diff_window})"