import numpy as np
import pandas as pd

class VixPanicExhaustionFactor:
    """波动率极值衰竭反转因子 (Volatility Microstructure / Options)

    逻辑: 捕捉极端流动性恐慌(Dash for Cash)见顶回落后的美债抄底机会。
          当股市发生极端恐慌时(VIX暴涨), 强迫去杠杆会导致全资产(包括美债)遭到无差别抛售。
          一旦这种期权微观结构层面的恐慌开始衰竭(VIX极值后回落), 
          强迫平仓结束, 资金将迅速回流避险资产(TLT)。
          这是一个典型的微观流动性脉冲, 必须作为狙击手级脉冲而非连续信号。
    数据: vixcls (CBOE VIX 波动率指数)
    触发: 极值条件 (VIX 252日 Z-Score > 2.5) + 衰竭条件 (VIX < 过去3日均值, 即恐慌开始退潮)
    输出: +1.0 (恐慌衰竭, 看多美债), 其余时间严格为 0.0
    """

    def __init__(self):
        self.name = 'vix_panic_exhaustion_pulse'
        self.zscore_lookback = 252    # 1个交易年的滚动窗口, 评估宏观波动率水位
        self.zscore_threshold = 2.5   # 统计学极值阈值 (右尾极端恐慌)
        self.exhaustion_window = 3    # 3日均线捕捉极短期的动量衰竭

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律: 初始信号必须为 0.0 (零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        # 处理数据缺失
        if 'vixcls' not in data.columns:
            signal.name = self.name
            return signal
            
        vix = data['vixcls'].ffill()
        
        # 计算 252 日滚动 Z-Score (宏观极值条件)
        vix_mean = vix.rolling(window=self.zscore_lookback, min_periods=self.zscore_lookback // 2).mean()
        vix_std = vix.rolling(window=self.zscore_lookback, min_periods=self.zscore_lookback // 2).std()
        
        # 避免除以 0 的情况
        vix_std = vix_std.replace(0, np.nan)
        z_score = (vix - vix_mean) / vix_std
        
        # 计算微观衰竭条件 (二阶导数铁律: 必须等恐慌开始回落)
        # vix < vix.rolling(3).mean() 等价于今天的 VIX 低于前两天的均值, 是绝佳的短线破位信号
        vix_exhaustion = vix < vix.rolling(window=self.exhaustion_window).mean()
        
        # 组合条件: 极端恐慌(接飞刀高危区) + 恐慌开始实质性回落(衰竭确认)
        trigger_condition = (z_score > self.zscore_threshold) & vix_exhaustion
        
        # 触发脉冲信号: 看多美债
        signal[trigger_condition] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_lookback={self.zscore_lookback}, zscore_threshold={self.zscore_threshold})"