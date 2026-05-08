import numpy as np
import pandas as pd

class UnstructuredNewsPanicReversalFactor:
    """新闻政策不确定性反转脉冲 (volatility/unstructured)

    逻辑: 每日新闻政策不确定性(EPU)基于大量非结构化文本计算，其极端飙升代表宏观叙事极度混乱(如危机爆发)。当这种恐慌到达极端高位且开始瓦解(伴随VIX同步回落)时，市场对不确定性的计价达到顶峰并走向平稳，此时往往是配置避险资产(美债)的绝佳右侧买点，避免主跌浪接飞刀。极度自满(EPU极低)且开始掉头回升时，常是通胀/加息冲击的起点，看空美债。
    数据: usepuindxd (基于新闻的经济政策不确定性指数), vixcls (VIX波动率指数)
    触发: 极值条件 (EPU 252日 Z-Score > 2.5) + 衰竭确认 (EPU 3日均值 diff() < 0) + 跨资产确认 (VIX diff() < 0)
    输出: +1.0 (恐慌衰竭看多脉冲), -1.0 (自满瓦解看空脉冲), 脉冲型
    """

    def __init__(self):
        self.name = 'unstructured_news_panic_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 初始信号全为0，保持零值休眠
        signal = pd.Series(0.0, index=data.index)
        
        # 检查数据完整性
        if 'usepuindxd' not in data.columns or 'vixcls' not in data.columns:
            return signal
            
        # 前向填充缺失值，避免未来函数
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        
        # 铁律3: 边际变化。对于日频新闻产生的噪音进行3日平滑，以提取真实的趋势变动
        epu_smooth = epu.rolling(window=3).mean()
        
        # 计算具有经济学含义的 252交易日(一年)滚动 Z-Score
        epu_roll_mean = epu_smooth.rolling(window=252).mean()
        epu_roll_std = epu_smooth.rolling(window=252).std()
        
        epu_zscore = (epu_smooth - epu_roll_mean) / (epu_roll_std + 1e-6)
        
        # 铁律2: 二阶导数 (变化动量)
        epu_diff = epu_smooth.diff()
        vix_diff = vix.diff()
        
        # 脉冲触发条件1: 恐慌极值 + 恐慌瓦解衰竭 + VIX跨资产回落确认
        long_condition = (epu_zscore > 2.5) & (epu_diff < 0) & (vix_diff < 0)
        
        # 脉冲触发条件2: 极度自满(低极值) + 恐慌开始抬头 + VIX抬头
        short_condition = (epu_zscore < -2.0) & (epu_diff > 0) & (vix_diff > 0)
        
        # 狙击手级脉冲赋值
        signal[long_condition] = 1.0
        signal[short_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"