import numpy as np
import pandas as pd

class UnstructuredEpuShockPulseFactor:
    """经济政策不确定性突变脉冲因子 (Unstructured NLP)

    逻辑: 经济政策不确定性(EPU, 基于新闻媒体文本挖掘)的突跳往往预示着宏观经济将承受巨大压力，迫使美联储降息以对冲风险(利多美债)。但黑天鹅事件初期的极度恐慌可能引发流动性危机导致股债双杀。因此必须等待不确定性的飙升达到极值且动量开始衰竭时，才形成高胜率的美债做多脉冲。反之，极度自满的破裂则触发看空脉冲。
    数据: usepuindxd (美国经济政策不确定性指数)
    触发: EPU的10日动量达到252日Z-Score > 2.5，且当日动量跌破3日均值(衰竭)。
    输出: 满足条件时触发 +1.0 或 -1.0 脉冲，其余时间严格为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_epu_shock_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 异常处理：如果没有所需字段
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 获取数据并处理缺失值
        epu = data['usepuindxd'].ffill()

        # 铁律3: 边际变化 (Marginal Change Only)
        # 计算 10日边际变化量，捕捉短期预期的阶跃突跳，同时抹平新闻日频噪音
        epu_momentum = epu.diff(10)

        # 计算 252日 Z-Score，用于定位极值事件
        rolling_mean = epu_momentum.rolling(window=252, min_periods=60).mean()
        rolling_std = epu_momentum.rolling(window=252, min_periods=60).std()
        
        # 避免极端情况下的除以 0 问题
        rolling_std = rolling_std.replace(0, np.nan).fillna(1e-5)
        z_score = (epu_momentum - rolling_mean) / rolling_std

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 计算动量的短期 3 日均值，用于捕捉加速势头的衰竭迹象
        epu_momentum_ma3 = epu_momentum.rolling(window=3).mean()
        
        # 衰竭条件判断：当前动量弱于最近3天的平均水平，说明单边势头正在破裂
        is_long_exhausted = epu_momentum < epu_momentum_ma3
        is_short_exhausted = epu_momentum > epu_momentum_ma3

        # 铁律1: 零值休眠 (Sniper Pulse)
        signal = pd.Series(0.0, index=data.index)

        # 看多脉冲：不确定性剧烈飙升 (利好避险降息) + 飙升势头开始衰竭
        long_trigger = (z_score > 2.5) & is_long_exhausted
        
        # 看空脉冲：不确定性剧烈下砸 (极度自满/过热) + 自满势头开始反弹
        short_trigger = (z_score < -2.5) & is_short_exhausted

        # 信号赋值
        signal.loc[long_trigger] = 1.0
        signal.loc[short_trigger] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"