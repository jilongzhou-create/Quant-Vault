import numpy as np
import pandas as pd

class VolatilityNewsUncertaintyReversalFactor:
    """新闻政策不确定性与波动率共振反转因子 (volatility/unstructured)

    逻辑: 结合非结构化新闻生成的经济政策不确定性(USEPUINDXD)与VIX。极端恐慌衰竭时做多美债(确认政策底与避险资金沉淀)，岁月静好时预期突发性恶化则做空(防范流动性冲击下的股债双杀)。
    数据: usepuindxd (基于新闻文本提取的每日经济不确定性), vixcls
    触发: 
      - 多头脉冲: EPU或VIX绝对水位 Z-Score > 2.5 且两者同步出现二阶导衰竭 (diff < 0)
      - 空头脉冲: 常态低位时 (Z < 0.5)，EPU或VIX的边际变化率出现极值跳升 (一阶导 Z-Score > 2.5)
    输出: [-1.0, 1.0] 的狙击手脉冲信号
    """

    def __init__(self):
        self.name = 'vol_news_uncertainty_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，默认全为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['usepuindxd', 'vixcls']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # 数据前向填充，防止低频/节假日缺失导致 NaN 影响对齐
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        
        # 1. 计算绝对水位的 252日 Z-Score
        epu_mean = epu.rolling(window=252, min_periods=126).mean()
        epu_std = epu.rolling(window=252, min_periods=126).std()
        epu_z = (epu - epu_mean) / (epu_std + 1e-8)
        
        vix_mean = vix.rolling(window=252, min_periods=126).mean()
        vix_std = vix.rolling(window=252, min_periods=126).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)
        
        # 2. 计算边际变化一阶导及其 Z-Score (铁律3)
        epu_diff = epu.diff()
        vix_diff = vix.diff()
        
        epu_diff_mean = epu_diff.rolling(window=252, min_periods=126).mean()
        epu_diff_std = epu_diff.rolling(window=252, min_periods=126).std()
        epu_diff_z = (epu_diff - epu_diff_mean) / (epu_diff_std + 1e-8)
        
        vix_diff_mean = vix_diff.rolling(window=252, min_periods=126).mean()
        vix_diff_std = vix_diff.rolling(window=252, min_periods=126).std()
        vix_diff_z = (vix_diff - vix_diff_mean) / (vix_diff_std + 1e-8)
        
        # 3. 铁律2: 二阶导数确认，抓极端恐慌的瓦解，防止接飞刀
        # 多头条件：
        # - 极值条件: 至少有一个宏观情绪指标处于极端狂热 (> 2.5)
        # - 确认条件: 跨指标互相确认，两者都必须脱离常态 (> 1.0)
        # - 衰竭条件: 恐慌开始退潮 (当天出现同步回落)
        is_extreme = (epu_z > 2.5) | (vix_z > 2.5)
        is_high = (epu_z > 1.0) & (vix_z > 1.0)
        is_exhausted = (epu_diff < 0) & (vix_diff < 0)
        
        long_cond = is_extreme & is_high & is_exhausted
        
        # 4. 铁律3: 边际变化突变跳升脉冲
        # 空头条件：
        # - 突变跳升: 某一个宏观情绪指标的日内变化幅度打出极值 (> 2.5)
        # - 常态被打破: 发生前市场必须是非高恐慌态 (绝对水位 Z < 0.5)，证明这是一次“意外冲击”
        is_surge = (epu_diff_z > 2.5) | (vix_diff_z > 2.5)
        is_quiet = (epu_z < 0.5) & (vix_z < 0.5)
        
        short_cond = is_surge & is_quiet
        
        # 脉冲赋值
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"