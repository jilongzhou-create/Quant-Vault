import numpy as np
import pandas as pd

class UnstructuredFomcVolExhaustionFactor:
    """央行沟通波动率衰竭脉冲 (volatility/unstructured)

    逻辑: 衡量美联储FOMC非结构化情绪得分的“沟通波动率”(Policy Communication Volatility)。当美联储在短期内鹰鸽立场剧烈摇摆(如前次极鹰、本次转鸽、下次又转鹰), 会导致情绪得分的波动率飙升, 市场为巨大的政策不确定性计入极高的期限溢价(抛售美债)。当这种沟通层面的混乱达到历史极值(Z-Score > 2.5)且最终开始衰竭(波动率回落)时, 标志着政策路径重新清晰, 不确定性溢价被挤出, 触发看多美债(TLT)的狙击脉冲。反之, 在极端死水期(波动率极低)突然产生分歧时看空美债。
    数据: fomc_sentiment (非结构化鹰鸽情绪得分)
    触发: 126日情绪波动率的 504日 Z-Score > 2.5 且 波动率边际回落(diff<0) -> +1.0
    输出: 狙击手级别的脉冲信号 [-1.0, 1.0], 常态下严格为 0.0
    """

    def __init__(self):
        self.name = 'unstructured_fomc_vol_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号为全 0.0 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)

        if 'fomc_sentiment' not in data.columns:
            return signal

        # 处理非结构化文本的情绪得分 (阶梯状数据前向填充)
        sentiment = data['fomc_sentiment'].ffill()

        # 1. 计算央行沟通的"波动率" 
        # 126个交易日约包含3次FOMC会议, 衡量近期的立场摇摆程度
        sentiment_vol = sentiment.rolling(window=126, min_periods=45).std()

        # 2. 波动率的极值度量 (Z-Score)
        # 使用过去约两年的窗口(504日)衡量当前的沟通波动率是否处于极端水平
        vol_mean = sentiment_vol.rolling(window=504, min_periods=126).mean()
        vol_std = sentiment_vol.rolling(window=504, min_periods=126).std()
        
        # 防止除以0产生无穷大
        vol_std = vol_std.replace(0, np.nan) 
        vol_z = (sentiment_vol - vol_mean) / vol_std

        # 3. 边际变化与衰竭条件 (二阶导数与边际变化铁律)
        # 绝对禁止波动率极端时盲目买入, 必须等波动率动量转折
        vol_diff = sentiment_vol.diff(5) # 5日边际变化, 捕捉会议落地后的阶跃
        vol_ma10 = sentiment_vol.rolling(window=10).mean()

        # 衰竭确认: 沟通波动率从高位回落, 且跌破近期均线 (不确定性消散)
        exhaustion = (vol_diff < 0) & (sentiment_vol < vol_ma10)

        # 爆发确认: 沟通波动率从死水期突然飙升 (不确定性重燃)
        acceleration = (vol_diff > 0) & (sentiment_vol > vol_ma10)

        # 触发做多脉冲 (+1.0): 沟通极度混乱后开始清晰, 期限溢价回落, 资金回流避险美债
        long_cond = (vol_z > 2.5) & exhaustion

        # 触发做空脉冲 (-1.0): 政策极度可预测(波动率极低)后突然出现分歧预期, 市场重定价引发抛售
        short_cond = (vol_z < -2.0) & acceleration

        # 仅在触发瞬间赋值, 其余时间维持 0.0
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"