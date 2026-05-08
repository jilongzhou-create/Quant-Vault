import numpy as np
import pandas as pd

class UnstructuredPolicyUncertaintyExhaustionFactor:
    """非结构化政策不确定性衰竭因子 (unstructured/unstructured)

    逻辑: 当美国经济政策不确定性指数(EPU)飙升至高位(Z>1.5)并开始衰竭回落时(单日下跌且跌破5日线), 说明短期恐慌冲击的靴子落地. 此时利用上一季度FOMC情绪得分的边际变化(动量)作为过滤: 若边际转鸽, 靴子落地往往催化衰退预期的兑现避险盘, 做多美债; 若边际转鹰, 则回落本质上是对政策紧缩预期的最终接受与消化, 做空美债. 该机制确保常态下处于零值休眠, 仅在不确定性退潮拐点输出狙击手脉冲信号.
    数据: usepuindxd, fomc_sentiment
    触发: usepuindxd的252日EWMA Z-Score > 1.5 且出现二阶导数衰竭(diff<0及破5日均线), 结合 fomc_sentiment.diff(63) 的边际方向定夺.
    输出: +1.0 (多) 或 -1.0 (空) 的离散脉冲
    """

    def __init__(self):
        self.name = 'unstructured_policy_uncertainty_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 必须验证所需数据是否在数据集中，若缺失直接返回 0.0 系列
        required_cols = ['usepuindxd', 'fomc_sentiment']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        epu = data['usepuindxd'].ffill()
        fomc = data['fomc_sentiment'].ffill()
        
        # =======================================================
        # 1. 铁律2: 二阶导数反转条件 (防止接飞刀)
        # =======================================================
        # 计算 EPU 的 EWMA 波动率和 Z-Score
        epu_mean = epu.ewm(span=252, min_periods=60).mean()
        epu_std = epu.ewm(span=252, min_periods=60).std().replace(0, np.nan).ffill()
        epu_z = (epu - epu_mean) / epu_std
        
        # 极值条件: Z-Score > 1.5 代表进入了较高政策不确定性的危机事件窗口
        is_extreme = epu_z > 1.5
        # 允许极值在过去3天内发生过, 用以捕捉波峰后初次回落
        extreme_recent = is_extreme.fillna(False).rolling(window=3).max() == 1
        
        # 衰竭条件 (核心二阶导数): 单日回落 且 跌破5日线确认动能退潮
        is_exhausted = (epu.diff(1) < 0) & (epu < epu.rolling(window=5).mean())
        
        # 靴子落地反转点 (不确定性消散的脉冲起点)
        trigger = extreme_recent & is_exhausted
        
        # =======================================================
        # 2. 铁律3: 边际变化限定 (禁止直接使用绝对值)
        # =======================================================
        # 63 个交易日约涵盖一个季度 (1~2次FOMC会议的间隔)
        # 通过季度的边际变化, 捕捉美联储预期的真实转向节奏, 不用0.5这种死板绝对值
        fomc_mom = fomc.diff(63)
        
        # =======================================================
        # 3. 铁律1: 零值休眠 (Sniper Pulse)
        # =======================================================
        # 只有在突发危机消散的关口, 结合当下的宏观主基调边际转变来触发一次脉冲
        cond_long = trigger & (fomc_mom > 0.05)   # 恐慌回落 + 整体边际在转鸽 = 确认看多美债
        cond_short = trigger & (fomc_mom < -0.05) # 恐慌回落 + 整体边际在转鹰 = 确认看空美债
        
        signal[cond_long] = 1.0
        signal[cond_short] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"