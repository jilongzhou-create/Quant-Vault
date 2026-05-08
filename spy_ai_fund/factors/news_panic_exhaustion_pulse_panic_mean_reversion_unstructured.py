import numpy as np
import pandas as pd

class NewsPanicExhaustionPulseFactor:
    """新闻恐慌极值与衰竭脉冲因子 (panic_mean_reversion/unstructured)

    逻辑: 基于新闻文本挖掘的经济政策不确定性(EPU)往往对股市产生压制。但美股具有长牛均值回归属性, 
          当新闻政策恐慌飙升至历史极值并开始边际衰竭时, 意味着"利空出尽", 触发强烈的看多脉冲。
          相反, 若政策不确定性在非极值区缓慢升温(温水煮青蛙), 则压制估值, 触发看空脉冲。
    数据: usepuindxd (基于新闻文本的每日美国经济政策不确定性指数)
    输出: +1.0 (新闻恐慌极值且衰竭, 抄底多头), -1.0 (不确定性温和恶化, 看空), 0.0 (常态休眠)
    触发条件: 126日Z-Score > 1.5 且今日回落破5日线触发 +1.0; 0.5 < Z < 1.5 且连续两日上升触发 -1.0。预期 Trigger Rate 约 8%-12%
    """

    def __init__(self, window: int = 126):
        self.name = 'news_panic_exhaustion_pulse'
        self.window = window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 默认返回全 0.0 的 Series
        signal = pd.Series(0.0, index=data.index)
        
        # 检查是否有所需的数据字段
        if 'usepuindxd' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        
        # 计算基于近半年的政策不确定性 Z-Score (评估所处的新闻恐慌水位)
        roll_mean = epu.rolling(window=self.window, min_periods=21).mean()
        roll_std = epu.rolling(window=self.window, min_periods=21).std()
        z_score = (epu - roll_mean) / (roll_std + 1e-6)
        
        # 极值特征：过去3日内曾经达到过高恐慌状态 (防接飞刀，允许冲高后稍作确认)
        high_panic_recently = z_score.rolling(window=3, min_periods=1).max() > 1.5
        
        # 均线与边际变化特征
        sma_5 = epu.rolling(window=5, min_periods=1).mean()
        sma_10 = epu.rolling(window=10, min_periods=1).mean()
        diff_1 = epu.diff()
        
        # 抄底信号 (+1.0): 处于极端恐慌区间，但今日出现衰竭 (低于周度均线且今日边际回落)
        # 符合二阶导数防飞刀铁律
        bull_cond = (
            high_panic_recently & 
            (epu < sma_5) & 
            (diff_1 < 0)
        )
        
        # 看空信号 (-1.0): 恐慌处于温和上升期 (未达到极度恐慌产生均值回归)，且连续两天新闻负面预期升温
        bear_cond = (
            (z_score > 0.5) & (z_score < 1.5) &
            (diff_1 > 0) & 
            (epu.shift(1).diff() > 0) & 
            (epu > sma_10)
        )
        
        # 合并信号
        signal.loc[bull_cond] = 1.0
        signal.loc[bear_cond] = -1.0
        
        # 边际变化铁律: 如果新闻指数未发生变动 (阶梯数据的死水期), 强制归零
        signal.loc[diff_1 == 0] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window})"