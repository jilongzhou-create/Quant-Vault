import numpy as np
import pandas as pd

class EpuPanicExhaustionPulseFactor:
    """经济政策不确定性衰竭极值因子 (panic_mean_reversion/unstructured)

    逻辑: 基于新闻文本挖掘计算的经济政策不确定性指数(EPU, usepuindxd)是典型的非结构化恐慌指标。
          当EPU处于历史极端高位(前15%)且开始边际回落时, 代表宏观政策的不确定性(如大选、贸易战、加息悬念等)出尽并被市场消化。
          此时风险偏好将迅速修复，触发强烈均值回归的抄底买入信号。
          当EPU处于中高位(60%-85%)且短期内持续缓慢上升时, 代表政策不确定性如同钝刀割肉，压制并消耗市场情绪，触发看空信号。
    数据: [usepuindxd] (美国经济政策不确定性指数，基于NLP文本分析)
    输出: [-1.0, 1.0] 1.0代表极端恐慌衰竭后的做多脉冲, -1.0代表轻度恐慌温和蔓延的看空脉冲
    触发条件: EPU处于过去252天的85分位数以上且今日环比下降触发+1.0; 处于60-85分位数且在短期上升趋势中触发-1.0。预期 Trigger Rate 约 10%-15%。
    """

    def __init__(self):
        self.name = 'epu_panic_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化默认全0信号 (狙击手常态休眠)
        signal = pd.Series(0.0, index=data.index, name=self.name)
        
        # 数据存在性校验
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 提取数据并处理缺失值
        epu = data['usepuindxd'].ffill()
        
        # 计算过去一年(252个交易日)的动态分布，min_periods=60以度过预热期
        rolling_85 = epu.rolling(window=252, min_periods=60).quantile(0.85)
        rolling_60 = epu.rolling(window=252, min_periods=60).quantile(0.60)
        
        # 计算边际变化与短期趋势 (边际变化铁律)
        epu_diff = epu.diff(1)
        ma5 = epu.rolling(window=5, min_periods=3).mean()
        ma10 = epu.rolling(window=10, min_periods=5).mean()
        
        # 1. 极端恐慌 + 衰竭 (买入脉冲, 二阶导数防飞刀铁律)
        # 条件: 政策不确定性处于年度极端高位(>85%), 但今日预期出现缓解(diff < 0)
        is_extreme_panic = epu > rolling_85
        is_exhausting = epu_diff < 0
        buy_cond = is_extreme_panic & is_exhausting
        
        # 2. 轻度恐慌 + 蔓延恶化 (看空脉冲)
        # 条件: 政策不确定性处于中高位(60%-85%), 且当下正在连续发酵(短期均线多头 + 今日边际恶化)
        is_mild_panic = (epu > rolling_60) & (epu <= rolling_85)
        is_worsening = epu_diff > 0
        uptrend = ma5 > ma10
        sell_cond = is_mild_panic & is_worsening & uptrend
        
        # 注入信号 (互斥逻辑)
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"