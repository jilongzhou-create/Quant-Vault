import numpy as np
import pandas as pd

class EpuNewsPanicExhaustionFactor:
    """新闻恐慌极值与衰竭因子 (panic_mean_reversion/unstructured)

    逻辑: 采用基于新闻文本提取的每日经济政策不确定性指数(EPU)。当EPU飙升至历史高位(新闻面极度恐慌)后开始回落(二阶导数为负)，表明恐慌情绪耗竭，触发S&P500长牛抄底脉冲；当市场处于长期无忧期(EPU极低)且突然爆发新闻恐慌跳升时，动能恶化，触发看空脉冲。
    数据: usepuindxd (Daily News-based Economic Policy Uncertainty Index)
    输出: +1.0 (极度不确定性见顶衰竭，强烈看多), -1.0 (平静期突发异变，看空), 0.0 (常态)
    触发条件: Z-Score>1.2且3日动量转负回落触发买入，持续低位后突发0.8标准差跳升触发卖出。目标Trigger Rate: 5%-15%。
    """

    def __init__(self):
        self.name = 'epu_news_panic_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查数据是否存在
        if 'usepuindxd' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].replace(0, np.nan).ffill()
        if epu.isna().all():
            return signal
            
        # 使用 log1p 平滑新闻指数极端的跳跃噪声
        log_epu = np.log1p(epu)
        
        # 采用252天(一年)滚动窗口计算基准线
        log_epu_mean = log_epu.rolling(window=252, min_periods=60).mean()
        log_epu_std = log_epu.rolling(window=252, min_periods=60).std()
        
        # 计算 Z-Score，并应用5日均线识别整体高位/低位状态
        epu_z = (log_epu - log_epu_mean) / log_epu_std
        epu_z_smooth = epu_z.rolling(window=5, min_periods=1).mean()
        
        # ==============================================================
        # 1. 抄底信号 (极值 + 衰竭)
        # ==============================================================
        # 条件A: 过去10天内存在极高政策不确定性 (Z-Score > 1.2，约对应前11%的尾部恐慌)
        recent_extreme_panic = (epu_z_smooth > 1.2).rolling(window=10, min_periods=1).max() > 0
        
        # 条件B: 恐慌开始实质性衰竭 (当日跌破10日均线，且近3日回落超过0.4个标准差)
        epu_ma10 = log_epu.rolling(window=10, min_periods=1).mean()
        panic_exhaustion = (log_epu < epu_ma10) & (log_epu.diff(3) < -0.4 * log_epu_std)
        
        # 脉冲化: 仅在衰竭条件首次满足的当天触发
        trigger_buy = recent_extreme_panic & panic_exhaustion & (~panic_exhaustion.shift(1).fillna(False))
        
        # ==============================================================
        # 2. 看空信号 (钝变恶化: 长平静期 + 突发恐慌)
        # ==============================================================
        # 条件A: 市场在过去21天(一个月)一直处于极低的新闻不确定性(Z-Score < 0.5)
        long_complacency = (epu_z_smooth < 0.5).rolling(window=21, min_periods=1).min() > 0
        
        # 条件B: 突发变故，3日内异动跳升超过0.8个标准差
        sudden_panic_spike = log_epu.diff(3) > 0.8 * log_epu_std
        
        # 脉冲化: 仅在平静期突发异动的首日触发
        trigger_sell = long_complacency.shift(1).fillna(False) & sudden_panic_spike & (~sudden_panic_spike.shift(1).fillna(False))
        
        # ==============================================================
        # 信号赋值
        # ==============================================================
        signal.loc[trigger_buy] = 1.0
        signal.loc[trigger_sell] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"