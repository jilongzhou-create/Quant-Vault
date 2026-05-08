import numpy as np
import pandas as pd

class FearGreedPanicReversionFactor:
    """CNN恐惧与贪婪极值回归因子 (panic_mean_reversion/unstructured)

    逻辑: 针对SPY长牛和均值回归特性，利用CNN恐惧与贪婪指数（广义另类情绪数据）捕捉市场的极致情绪极值与反转。根据"极度恐慌产生买点但绝不接飞刀"的物理法则，当指数处于极度恐惧区间（<20）且日内边际反弹时，说明恐慌明确衰竭，触发抄底看多信号；相反，当指数处于常态区间（40~60）却连续缓慢下行，说明资金在温水煮青蛙式撤离，没有触发极值的慢性恶化意味着单边阴跌，此时看空美股。
    数据: [fear_greed]
    输出: +1.0 (恐慌极值衰竭看多), -1.0 (轻度恐慌阴跌发酵看空), 0.0 (常态休眠)
    触发条件: 过去3日最低值<20且今日动量反弹时+1.0；绝对值在40-60之间且连续3日动量为负时-1.0。预期 Trigger Rate 控制在 5% 到 15% 之间。
    """

    def __init__(self):
        self.name = 'fear_greed_panic_reversion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 数据缺失保护防御
        if 'fear_greed' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        # 提取指标并处理前向填充
        fg = data['fear_greed'].ffill()
        signal = pd.Series(0.0, index=data.index)
        
        # ---------------------------------------------
        # 抄底看多逻辑 (防接飞刀极值回归法则)
        # ---------------------------------------------
        # 前置条件: 过去3日内，恐惧贪婪指数曾跌破20，步入"极度恐惧"的黑天鹅或巨幅回调区间
        extreme_panic = fg.rolling(window=3).min() < 20
        # 衰竭条件: 绝对禁止直接抄底！必须等待恐慌边际减弱（今日恐惧值较昨日开始上升回暖）
        exhaustion = fg.diff() > 0
        
        # ---------------------------------------------
        # 趋势恶化看空逻辑 (钝刀子割肉阴跌法则)
        # ---------------------------------------------
        # 前置条件: 情绪并未跌入极值引发超跌反弹，而是处于中性区间 (40 ~ 60)
        neutral_zone = (fg >= 40) & (fg <= 60)
        # 衰退条件: 情绪连续3个交易日呈下滑状态，反映预期缓慢变差，主跌浪极可能刚刚形成
        slow_deterioration = (fg.diff() < 0) & \
                             (fg.shift(1).diff() < 0) & \
                             (fg.shift(2).diff() < 0)
                             
        # 信号合成与互斥逻辑
        buy_signal = extreme_panic & exhaustion
        sell_signal = neutral_zone & slow_deterioration
        
        # 输出脉冲信号 (+1.0 或 -1.0)
        signal[buy_signal] = 1.0
        signal[sell_signal] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"