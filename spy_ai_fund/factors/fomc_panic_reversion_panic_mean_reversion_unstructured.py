import numpy as np
import pandas as pd

class FomcPanicReversionFactor:
    """FOMC情绪边际重估与恐慌衰竭脉冲 (panic_mean_reversion/unstructured)

    逻辑: 通过NLP解析的FOMC鹰鸽情绪打分(fomc_sentiment)，捕捉低频政策阶梯数据的边际突变。
          美股是一个均值回归属性极强的长牛市场，做多绝佳机会往往出现在"恐慌见顶衰竭"的瞬间：
          即前期一直处于鹰派紧缩预期(负值恐慌)，随后某次议息会议意外边际转鸽(二阶变化>0)，
          此"衰竭瞬间"触发强烈看多脉冲。反之，只要会议边际转鹰(<0)，则是流动性退潮趋势恶化，果断给出防守看空信号。
    数据: [fomc_sentiment]
    输出: +1.0 看多(恐慌衰竭，紧缩见顶)；-1.0 看空(转鹰或紧缩持续恶化)
    触发条件: 会议文本情绪出现显著变化瞬间及随后的3天内(构成4天脉冲)，预期 Trigger Rate 控制在 8% - 15% 之间
    """

    def __init__(self):
        self.name = 'fomc_panic_reversion_unstructured'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 缺列处理
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        # 阶梯状情绪数据：T+1生效，非会议日前向填充
        fomc = data['fomc_sentiment'].ffill()
        
        # 【边际变化铁律】: 严禁对阶梯数据直接输出绝对值，必须使用动量突变
        fomc_diff = fomc.diff()
        prev_fomc = fomc.shift(1)
        
        # 【二阶导数与均值回归法则】
        # 1. 恐慌见顶与衰竭脉冲 (抄底买点):
        # 条件：前期处于紧缩鹰派压制下(prev < 0.0)，且本次会议出现边际转鸽的明显跳跃(diff >= 0.05)
        # 或者本次会议直接展现出大超预期的超级鸽派突变(diff >= 0.15)
        bull_trigger = ((prev_fomc < 0.0) & (fomc_diff >= 0.05)) | (fomc_diff >= 0.15)
        
        # 2. 趋势恶化与钝刀割肉脉冲 (避险卖点):
        # 条件：只要FOMC政策路径边际恶化(转鹰幅度 <= -0.05)，不管之前是鸽是鹰，皆为流动性负面冲击
        bear_trigger = fomc_diff <= -0.05
        
        # 确保布尔序列的安全填充
        bull_trigger = bull_trigger.fillna(False).astype(bool)
        bear_trigger = bear_trigger.fillna(False).astype(bool)
        
        # 【零值休眠铁律】
        # 仅仅在重估跳跃发生的当天及随后极短几天内给出信号。
        # 每年约8次会议及纪要带来情绪阶梯，将瞬间突变延展为 4 天窗口(当天+3天消化期)
        # 预期每年总信号天数在 20 - 40 天左右，确保 Trigger Rate 介于 5% - 15%
        pulse_window = 4
        bull_pulse = bull_trigger.rolling(window=pulse_window, min_periods=1).max().fillna(0).astype(bool)
        bear_pulse = bear_trigger.rolling(window=pulse_window, min_periods=1).max().fillna(0).astype(bool)
        
        # 拼装最终信号
        signal = pd.Series(0.0, index=data.index)
        signal[bull_pulse] = 1.0
        signal[bear_pulse] = -1.0
        
        # 防护逻辑：重叠发生时(理论极少)，中和归零
        conflict = bull_pulse & bear_pulse
        signal[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"