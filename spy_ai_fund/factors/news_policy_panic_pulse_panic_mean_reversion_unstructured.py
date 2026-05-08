import numpy as np
import pandas as pd

class NewsPolicyPanicPulseFactor:
    """News Policy Panic Pulse Factor (panic_mean_reversion/unstructured)

    逻辑: 每日经济政策不确定性指数(usepuindxd, 基于新闻NLP提取)达到极端高位后拐头向下(恐慌极值+衰竭), 触发看多脉冲; 单日突发不确定性飙升触发看空脉冲。该因子精准捕捉基于非结构化新闻文本量化的政策恐慌情绪。
    数据: [usepuindxd]
    输出: 政策恐慌衰竭瞬间输出+1.0(看多美股), 突发政策冲击瞬间输出-1.0(趋势恶化看空), 其余时间0.0
    触发条件: 252日Z-Score > 1.25 且动量由正转负触发买入脉冲; Z-Score > 1.0 且单日飙升 > 0.8 触发卖出脉冲。预期 Trigger Rate 约 6%-10%。
    """

    def __init__(self):
        self.name = 'news_policy_panic_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 缺失字段处理: 如果所需的 usepuindxd 不存在, 直接返回全 0
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        epu = data['usepuindxd'].ffill()
        
        # 计算基于一年的滚动 Z-Score (代表长牛市的常态基准)
        epu_mean = epu.rolling(window=252, min_periods=63).mean()
        epu_std = epu.rolling(window=252, min_periods=63).std().replace(0, np.nan)
        epu_z = (epu - epu_mean) / epu_std
        
        # 【二阶导数铁律 & 极值衰竭】 
        # 1. 抄底买入脉冲 (恐慌极值 + 见顶回落的瞬间)
        # 条件: 昨天不确定性处于历史高位 (Z > 1.25)
        # 且昨天是上升的(恐慌加剧), 今天下跌(恐慌开始衰竭, 预期反转)
        buy_cond = (
            (epu_z.shift(1) > 1.25) & 
            (epu.diff() < 0) & 
            (epu.diff().shift(1) > 0)
        )
        
        # 【边际变化铁律】
        # 2. 趋势恶化卖出脉冲 (突发恐慌冲击)
        # 条件: 今天不确定性突然跳涨 (Z-score 单日剧烈变化 > 0.8)
        # 且今天绝对水平已处于较高恐慌状态 (Z > 1.0)
        sell_cond = (
            (epu_z > 1.0) & 
            (epu_z.diff() > 0.8)
        )
        
        # 初始化零值休眠序列
        signal = pd.Series(0.0, index=data.index)
        
        # 赋值脉冲
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        # 防止同一天冲突（极罕见），保守处理：如果同日发生，置0
        conflict = buy_cond & sell_cond
        signal[conflict] = 0.0
        
        # 确保数据初期的历史计算窗口内不乱发信号
        signal[:63] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"