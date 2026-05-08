import numpy as np
import pandas as pd

class UnstructuredEpuVolatilityReversalFactor:
    """不确定性波动极值与衰竭反转因子 (volatility/unstructured)

    逻辑: 监控非结构化新闻文本衍生的经济政策不确定性指数(USEPUINDXD)。不确定性极端飙升代表恐慌和流动性收紧危机，美债往往遭遇无差别抛售；当其见顶并跌破短期均线的瞬间(靴子落地/恐慌衰竭)，预期央行干预或避险资金回流，触发看多美债的反转脉冲。反之，当不确定性长期处于极低冰点(市场极度自满狂欢)，一旦突然向上跳升突破均线打破平静(通胀或紧缩担忧初现)，长端美债往往因加息预期升温而迅速下跌，触发看空脉冲。因子严格遵循零值休眠与二阶导数反转铁律。
    数据: usepuindxd (经济政策不确定性指数)
    触发: 看多 = 过去5日内 EPU Z-Score > 2.2，且今日有效下穿5日均线(边际衰竭)；看空 = 过去5日内 EPU Z-Score < -1.8，且今日有效上穿5日均线(边际跳升)。
    输出: 反转跳跃当天的狙击手级脉冲信号 (+1.0 或 -1.0)，常态为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_epu_volatility_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1 (零值休眠): 初始信号必须全为 0.0，常态静默
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 前向填充缺失值
        epu = data['usepuindxd'].ffill()
        
        # 计算 252 个交易日（约一年）的滚动均值和标准差，用于评估宏观状态的极端偏离
        window = 252
        epu_mean = epu.rolling(window=window).mean()
        # 替换 std 为 0 的情况防止除零错误
        epu_std = epu.rolling(window=window).std().replace(0, np.nan)
        zscore = (epu - epu_mean) / epu_std
        
        # 计算短期均线，用于捕捉衰竭拐点（边际变化确认）
        epu_ma5 = epu.rolling(window=5).mean()
        
        # =========================================================
        # 脉冲触发逻辑设计 (绝对禁止无脑买入，严格执行极值+衰竭二阶导数)
        # =========================================================
        
        # 1. 看多脉冲：不确定性狂飙到极点后发生向下死叉
        # 铁律2 (Anti-Catch-Falling-Knife): 必须同时满足 极值条件(曾摸到2.2σ以上) 和 衰竭条件(死叉回落)
        recent_high = zscore.rolling(window=5).max() > 2.2
        # 寻找真正的拐点瞬间：昨天还在均线上方，今天有效跌破均线且单日下挫
        cross_down = (epu.shift(1) >= epu_ma5.shift(1)) & (epu < epu_ma5) & (epu.diff() < 0)
        long_cond = recent_high & cross_down
        
        # 2. 看空脉冲：不确定性长期极度低迷后发生向上金叉
        # 铁律3 (边际变化): 绝对不用低位的绝对连续值输出信号，而是捕捉平静被打破、均线被向上击穿的阶跃突变瞬间
        recent_low = zscore.rolling(window=5).min() < -1.8
        # 寻找向上跳跃的突变点：昨日被压制，今日猛烈抬头并突破均线
        cross_up = (epu.shift(1) <= epu_ma5.shift(1)) & (epu > epu_ma5) & (epu.diff() > 0)
        short_cond = recent_low & cross_up
        
        # 仅在触发时离散赋值，满足 5%~15% 的狙击手级 Trigger Rate
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"