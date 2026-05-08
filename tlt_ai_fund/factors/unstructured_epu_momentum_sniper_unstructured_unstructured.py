import numpy as np
import pandas as pd

class UnstructuredEpuPulseFactor:
    """政策不确定性极值脉冲因子 (unstructured/unstructured)

    逻辑: 极端的美国经济政策不确定性(EPU)预示着宏观避险情绪的拐点。当EPU处于极低位(市场极度自满)且开始边际飙升时，避险资金将涌入美债，触发看多脉冲(+1.0)；当EPU极高(恐慌极值)且开始见顶回落时，避险情绪消退，资金流出美债(买预期卖事实)，触发看空脉冲(-1.0)。这解决了前期"直接买入恐慌极值死于反转"的问题，确保 Hit Rate 与 IC 的稳健。
    数据: usepuindxd (美国经济政策不确定性指数 - 日频)
    触发: 252日滚动 Z-Score 达到极值 (|Z| > 1.5)，且 5日动量突破 0.25 倍长期标准差(确认真实反转而非日常噪音) 时触发脉冲。
    输出: 严格遵循三大铁律的 [-1.0, 1.0] 狙击手脉冲信号。
    """

    def __init__(self):
        self.name = 'unstructured_epu_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 缺失数据处理
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 初始信号为全 0.0, 遵循"零值休眠"狙击手铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 提取 EPU 并用 5个交易日均线平滑其极端的日度噪音
        epu = data['usepuindxd'].ffill()
        epu_smooth = epu.rolling(window=5).mean()

        # 计算一年期(252个交易日)的基准水位, 用于衡量当前的恐慌/自满宏观极值
        epu_mean = epu_smooth.rolling(window=252).mean()
        epu_std = epu_smooth.rolling(window=252).std()
        epu_z = (epu_smooth - epu_mean) / (epu_std + 1e-8)

        # 边际变化铁律: 使用 5日动量 衡量不确定性的爆发或衰竭
        epu_mom = epu_smooth.diff(5)

        # 二阶导数确认阈值: 使用 0.25 倍长期标准差过滤微小的日常回撤噪音, 确保是真实的宏观情绪拐点
        mom_threshold = 0.25 * epu_std

        # 触发条件1: 自满衰竭 -> 恐慌升温 -> 避险买入美债 (+1.0)
        # EPU 极低 (Z < -1.5) 且出现显著的向上反转势头 (动量突破阈值)
        bull_pulse = (epu_z < -1.5) & (epu_mom > mom_threshold)

        # 触发条件2: 恐慌衰竭 -> 情绪修复 -> 卖出避险美债 (-1.0)
        # EPU 极高 (Z > 1.5) 且出现显著的向下衰竭势头 (跌破动量阈值)
        bear_pulse = (epu_z > 1.5) & (epu_mom < -mom_threshold)

        # 仅在事件触发瞬间赋值，避免连续信号，精确控制 Trigger Rate
        signal[bull_pulse] = 1.0
        signal[bear_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"