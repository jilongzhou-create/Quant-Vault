import numpy as np
import pandas as pd

class MicrostructureStressExhaustionFactor:
    """微观结构流动性压力衰竭因子 (microstructure/unstructured)

    逻辑: 金融压力(stlfsi4)反映市场微观流动性摩擦。当压力水位极高并开始边际回落时，标志恐慌挤兑见顶，避险资金重返美债，触发看多脉冲。反之，若压力突发极端跳升，引发无差别流动性抛售，触发看空脉冲。
    数据: stlfsi4 (圣路易斯联储金融压力指数)
    触发: 压力水位 Z-Score > 2.0 且边际回落(diff < 0) 看多；单日压力增量 Z-Score > 2.5 看空。触发后平滑维持3天。
    输出: +1.0(看多美债), -1.0(看空美债), 脉冲型
    """

    def __init__(self):
        self.name = 'microstructure_stress_exhaustion_unstructured'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理缺失字段
        if 'stlfsi4' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 获取数据并前向填充缺失值
        stress = data['stlfsi4'].ffill()
        
        # 计算 252 交易日滚动均值与标准差以确定压力水位极值
        roll_mean = stress.rolling(window=252, min_periods=60).mean()
        roll_std = stress.rolling(window=252, min_periods=60).std()
        stress_zscore = (stress - roll_mean) / (roll_std + 1e-6)
        
        # 铁律3: 阶梯型数据必须使用边际变化 (差分)
        stress_diff = stress.diff()
        
        # 对差分进行滚动标准化，用于捕捉"突然的流动性收紧跳升"
        diff_mean = stress_diff.rolling(window=252, min_periods=60).mean()
        diff_std = stress_diff.rolling(window=252, min_periods=60).std()
        diff_zscore = (stress_diff - diff_mean) / (diff_std + 1e-6)

        # 铁律1: 零值休眠
        signal = pd.Series(0.0, index=data.index)

        # 铁律2: 二阶导数抄底逻辑 (指标极端高位 + 边际回落)
        # 捕捉恐慌衰竭点，看多美债 (流动性挤兑结束, TLT 反弹)
        buy_cond = (stress_zscore.shift(1) > 2.0) & (stress_diff < 0)

        # 捕捉流动性突发紧缩黑天鹅，市场无差别抛售一切 (现金为王)，引发急跌看空
        sell_cond = (diff_zscore > 2.5) & (stress_diff > 0)

        signal.loc[buy_cond] = 1.0
        signal.loc[sell_cond] = -1.0

        # 铁律1补充: 极端事件发生的当天及随后极短几天内输出非零值
        # 展宽2天（总共持续3天）以适当维持脉冲，并将 Trigger Rate 调整至 5%~15% 区间
        signal = signal.replace(0.0, np.nan).ffill(limit=2).fillna(0.0)

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"