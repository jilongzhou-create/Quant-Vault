import numpy as np
import pandas as pd

class FomcSentimentPanicReversionFactor:
    """FOMC情绪极值突变与恐慌衰竭因子 (panic_mean_reversion/unstructured)

    逻辑: 捕捉美联储政策预期的极端反转瞬间。当市场处于鹰派恐慌(前次得分<0)且边际情绪大幅转鸽(diff>0.25)时, 标志政策紧缩恐慌衰竭, 输出看多脉冲抄底; 当市场处于鸽派幻觉(前次得分>0)且突然转鹰(diff<-0.25)时, 标志利好出尽, 输出看空脉冲。信号只在跳变发生当日及随后几天内存活。
    数据: [fomc_sentiment]
    输出: [-1.0, 1.0], 1.0=极度鹰派恐慌衰竭看多, -1.0=极度鸽派预期破灭看空, 0.0=常态休眠
    触发条件: 情绪分数发生跳跃且前期处于相应的极值状态, 脉冲维持5天。预期 Trigger Rate 在 8%-12% 之间。
    """

    def __init__(self, hawkish_extreme=-0.1, dovish_extreme=0.1, dovish_shift_threshold=0.25, hawkish_shift_threshold=-0.25, pulse_hold_days=5):
        self.name = 'fomc_sentiment_panic_reversion'
        self.hawkish_extreme = hawkish_extreme
        self.dovish_extreme = dovish_extreme
        self.dovish_shift_threshold = dovish_shift_threshold
        self.hawkish_shift_threshold = hawkish_shift_threshold
        self.pulse_hold_days = pulse_hold_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 确保阶梯状数据的前向填充
        sentiment = data['fomc_sentiment'].ffill()
        
        # 计算低频阶梯数据的边际变化 (脉冲识别)
        sentiment_diff = sentiment.diff()
        prev_sentiment = sentiment.shift(1)
        
        # 狙击看多脉冲: 鹰派恐慌见顶并开始衰竭 (前期偏鹰, 且最新声明大幅转鸽)
        # 或者出现史诗级的鸽派突变(不论前值)
        bullish_pulse = (
            ((prev_sentiment < self.hawkish_extreme) & (sentiment_diff >= self.dovish_shift_threshold)) | 
            (sentiment_diff >= self.dovish_shift_threshold + 0.15)
        )
        
        # 狙击看空脉冲: 鸽派幻觉破灭 (前期偏鸽, 且最新声明大幅转鹰)
        # 或者出现史诗级的鹰派突变(不论前值)
        bearish_pulse = (
            ((prev_sentiment > self.dovish_extreme) & (sentiment_diff <= self.hawkish_shift_threshold)) | 
            (sentiment_diff <= self.hawkish_shift_threshold - 0.15)
        )
        
        # 生成单日突变脉冲
        raw_signal = pd.Series(0.0, index=data.index)
        raw_signal.loc[bullish_pulse] = 1.0
        raw_signal.loc[bearish_pulse] = -1.0
        
        # 为了保证触发率(Trigger Rate)在 5%-15% 的黄金区间, 
        # 将跳变当天的脉冲向后维持极短天数 (pulse_hold_days = 5)
        # 单独滚动正负脉冲, 避免相互抵消或被 rolling max 掩盖负值
        pulse_bull = raw_signal[raw_signal > 0].reindex(data.index).fillna(0.0)
        pulse_bear = raw_signal[raw_signal < 0].reindex(data.index).fillna(0.0)
        
        bull_signal = pulse_bull.rolling(window=self.pulse_hold_days, min_periods=1).max()
        bear_signal = pulse_bear.rolling(window=self.pulse_hold_days, min_periods=1).min()
        
        # 融合多空信号
        signal = bull_signal + bear_signal
        
        # 清理异常并确保 [-1.0, 1.0] 界限
        signal = signal.clip(-1.0, 1.0).fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(pulse_hold_days={self.pulse_hold_days})"