import numpy as np
import pandas as pd

class EpuPanicExhaustionFactor:
    """新闻恐慌突变与极值衰竭因子 (panic_mean_reversion/unstructured)

    逻辑: 
    1. 抄底信号(+1.0): 每日经济政策不确定性指数(EPU)基于海量新闻文本衡量市场宏观恐慌。SPY长牛法则下，当EPU升至极端高位(Z>1.2)并出现二阶向下拐点时，标志着宏观恐慌情绪见顶衰竭，此时产生强力看多脉冲。
    2. 看空信号(-1.0): 仅当原始EPU出现单日爆发现象(Z-score日内飙升>2.0)，代表市场突发纯粹的黑天鹅级别利空(如意外地缘冲突)，触发短线避险看空脉冲。
    数据: usepuindxd (美国每日经济政策不确定性指数)
    输出: [-1.0, 1.0] 的脉冲信号
    触发条件: 平滑Z-Score>1.2且出现向下拐点触发买入(展期3天)；单日飙升>2.0触发卖出(展期2天)。预期 Trigger Rate 8%-12%。
    """
    
    def __init__(self):
        self.name = 'epu_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 如果缺少所需数据，直接返回常态零值
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        epu = data['usepuindxd'].ffill()
        
        # 长周期宏观基准 (126个交易日，约半年)
        epu_mean_126 = epu.rolling(window=126).mean()
        epu_std_126 = epu.rolling(window=126).std()
        
        # 1. 原始数据的 Z-Score，专门用于捕捉单日巨大突变 (防范短线暴跌)
        epu_raw_z = (epu - epu_mean_126) / (epu_std_126 + 1e-6)
        
        # 2. 经过 3日均线 平滑的 EPU，专门用于过滤日常新闻噪音，准确捕捉趋势拐点
        epu_ma3 = epu.rolling(window=3).mean()
        epu_ma3_z = (epu_ma3 - epu_mean_126) / (epu_std_126 + 1e-6)
        
        # --- 空头脉冲 (恐慌突发黑天鹅激增) ---
        # 原始 Z-score 单日飙升超过 2 个标准差，且绝对值处于恐慌水位
        shock_event = (epu_raw_z.diff(1) > 2.0) & (epu_raw_z > 1.5)
        # 突发恐慌冲击极强，维持 2 天抛售窗口
        bear_pulse = shock_event.rolling(window=2, min_periods=1).max() > 0
        
        # --- 多头脉冲 (恐慌极值 + 衰竭确认) ---
        # 条件1: 处于极度恐慌区间 (Z > 1.2)
        extreme_panic = epu_ma3_z > 1.2
        # 条件2: 二阶导数向下衰竭 (MA3今天实质性下降，且昨天是上升或持平的局部高点)
        turning_down = (epu_ma3 < epu_ma3.shift(1)) & (epu_ma3.shift(1) >= epu_ma3.shift(2))
        # 触发买入极值衰竭事件
        bull_event = extreme_panic & turning_down
        # 恐慌回落后的均值回归动能强，维持 3 天买入窗口
        bull_pulse = bull_event.rolling(window=3, min_periods=1).max() > 0
        
        # --- 信号合成 ---
        signal = pd.Series(0.0, index=data.index)
        
        # 先打入看空信号
        signal[bear_pulse] = -1.0
        # 后打入看多信号 (如果同一天既暴涨又见顶回落，优先尊重恐慌衰竭的长牛反转属性)
        signal[bull_pulse] = 1.0  
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(direction='panic_mean_reversion', method='unstructured')"