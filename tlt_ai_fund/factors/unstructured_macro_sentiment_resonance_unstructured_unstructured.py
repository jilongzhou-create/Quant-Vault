import numpy as np
import pandas as pd

class UnstructuredMacroSentimentResonanceFactor:
    """非结构化宏观情绪共振因子 (unstructured/unstructured)

    逻辑: 结合新闻经济政策不确定性(EPU)的极值反转与央行FOMC情绪的边际变化。当市场新闻恐慌飙升至极端且开始退潮降温，同时近一个月美联储态度发生鸽派转向时，预期差修复带来美债做多脉冲；反之做空。
    数据: usepuindxd (经济政策不确定性指数), fomc_sentiment (FOMC文本鹰鸽得分)
    触发: EPU 5日变化Z-Score > 2.5 且 3日内开始回落(二阶衰竭), 并且 FOMC 21日边际动量 > 0
    输出: +1.0 看多TLT, -1.0 看空TLT, 常态输出 0.0, 属于狙击手脉冲信号
    """

    def __init__(self):
        self.name = 'unstructured_macro_sentiment_resonance'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全零 Series，严格遵守常态 0.0 铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 必须处理数据缺失情况
        if 'usepuindxd' not in data.columns or 'fomc_sentiment' not in data.columns:
            return signal
            
        # 提取数据并前向填充防止缺失空洞
        epu = data['usepuindxd'].ffill()
        fomc = data['fomc_sentiment'].ffill()
        
        # 1. 边际变化计算 (遵守边际变化铁律)
        # EPU 5日动量: 反映一周内新闻情绪的发酵
        epu_diff = epu.diff(5)
        
        # FOMC 21日动量: 反映近一个月内央行态度的边际变化
        # 绝对禁止使用 fomc_sentiment 的阶梯绝对值!
        fomc_diff = fomc.diff(21)
        
        # 2. 极值监控 Z-Score (约252交易日窗口，即1年宏观记忆)
        epu_diff_mean = epu_diff.rolling(window=252, min_periods=60).mean()
        epu_diff_std = epu_diff.rolling(window=252, min_periods=60).std()
        
        # 安全除法
        epu_diff_std = epu_diff_std.replace(0, np.nan)
        epu_zscore = (epu_diff - epu_diff_mean) / epu_diff_std
        
        # 3. 二阶导数/衰竭条件 (反接飞刀铁律)
        # 恐慌开始降温: 过去3日不确定性边际回落
        epu_cooling = epu.diff(3) < 0
        # 乐观开始反弹: 过去3日不确定性边际回升
        epu_heating = epu.diff(3) > 0
        
        # 4. 复合脉冲触发逻辑
        # 鸽派避险脉冲 (+1.0)
        # 条件1: 动量 Z-Score > 2.5 (极度恐慌飙升)
        # 条件2: 恐慌衰竭 (EPU Cooling)
        # 条件3: 联储转向安抚 (FOMC Diff > 0)
        dove_pulse = (epu_zscore > 2.5) & epu_cooling & (fomc_diff > 0)
        
        # 鹰派紧缩脉冲 (-1.0)
        # 条件1: 动量 Z-Score < -2.5 (恐慌大幅骤降, 极度自满)
        # 条件2: 反转发酵 (EPU Heating)
        # 条件3: 联储转向打压 (FOMC Diff < 0)
        hawk_pulse = (epu_zscore < -2.5) & epu_heating & (fomc_diff < 0)
        
        # 5. 狙击手脉冲赋值
        signal[dove_pulse] = 1.0
        signal[hawk_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"