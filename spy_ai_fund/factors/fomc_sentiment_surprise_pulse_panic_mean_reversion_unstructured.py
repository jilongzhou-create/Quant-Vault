import numpy as np
import pandas as pd

class FomcSentimentSurprisePulseFactor:
    """FOMC非结构化情绪超预期脉冲因子 (panic_mean_reversion/unstructured)

    逻辑: 捕捉美联储货币政策预期发生边际转向的瞬间。基于美股长牛和对流动性极其敏感的属性，
          当FOMC声明情绪得分发生明显的鸽派突变(diff > 0.3)或绝对转向(由鹰转鸽)时，
          这意味着流动性紧缩恐慌正式"见顶衰竭"，随之而来的是强烈的估值修复，输出强看多脉冲。
          反之，当情绪出现鹰派惊吓(diff < -0.3)或由鸽转鹰时，打破了宽松自满状态，输出看空脉冲。
    数据: [fomc_sentiment]
    输出: [-1.0, 1.0] 的脉冲信号。常态极度静默为0.0，只在宏观情绪预期的转折日爆发。
    触发条件: 严格在FOMC情绪得分跳变的当天及随后3个交易日(极短窗口)内触发，预期Trigger Rate在8%-12%。
    """

    def __init__(self):
        self.name = 'fomc_sentiment_surprise_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1 & 数据保护: 如果核心数据缺失，直接返回全0序列
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)

        s = data['fomc_sentiment']
        
        # 铁律8 (边际变化铁律): 绝对禁止使用阶梯数据的绝对值，必须计算动量
        diff = s.diff()
        prev = s.shift(1)
        
        # 捕捉预期发生改变的瞬间
        jump_day = (diff != 0.0) & (diff.notna())
        
        # 初始化常态为 0.0 (狙击手休眠态)
        pulse = pd.Series(0.0, index=data.index)
        
        # 鸽派释放 (恐慌衰竭买点): 突变幅度 > 0.3，或彻底跨越零轴由鹰转鸽
        bull_cond = jump_day & ((diff > 0.3) | ((prev < 0.0) & (s > 0.0)))
        
        # 鹰派惊吓 (流动性自满恶化看空): 突变幅度 < -0.3，或由鸽转鹰
        bear_cond = jump_day & ((diff < -0.3) | ((prev > 0.0) & (s < 0.0)))
        
        pulse.loc[bull_cond] = 1.0
        pulse.loc[bear_cond] = -1.0
        
        # 铁律6 (零值休眠与Trigger Rate要求): 
        # 因为FOMC一年仅约8次，若只在当天触发，概率不到3.5%，无法达标。
        # 按照"在极端事件发生的当天及随后极短几天内"的原则，将脉冲顺延3个交易日。
        # 最大触发天数 8次 * 4天 = 32天 (最高12.7%的Trigger Rate)，完美锁定在 5%-15% 区间。
        pulse_sparse = pulse.replace(0.0, np.nan)
        pulse_extended = pulse_sparse.ffill(limit=3)
        
        signal = pulse_extended.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"