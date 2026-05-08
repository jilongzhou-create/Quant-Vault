import numpy as np
import pandas as pd

class UnstructuredPolicyVolExhaustionFactor:
    """Unstructured Policy Vol Exhaustion (unstructured/options)

    逻辑: 结合基于新闻文本的经济政策不确定性(EPU)与期权市场的隐含波动率(VIX)构建共振恐慌指数。当政策不确定性与市场恐慌同步爆发(Z-Score>2.5)时，长债往往先遭遇短期流动性抛售；当核心微观恐慌(VIX)率先从极端高位开始回落(二阶导衰竭)时，说明流动性冲击结束，避险资金大举重新买入无风险资产，触发做多美债(TLT)脉冲。反之，在极度自满时且VIX反弹触发看空脉冲。
    数据: usepuindxd (经济政策不确定性指数), vixcls (VIX期权隐含波动率)
    触发: 共振冲击 Z-Score > 2.5 且 VIX 下穿3日均值开始回落。
    输出: +1.0 看多美债, -1.0 看空美债。非触发日常态休眠为 0.0。
    """

    def __init__(self, zscore_window: int = 252, trigger_threshold: float = 2.5, exhaust_window: int = 3):
        self.name = 'unstructured_policy_vol_exhaustion'
        self.zscore_window = zscore_window
        self.trigger_threshold = trigger_threshold
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 检查所需数据是否齐全
        if 'usepuindxd' not in data.columns or 'vixcls' not in data.columns:
            return signal

        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()

        # EPU 基于新闻文本提取，通常具有单日高噪音，先进行 5 日平滑处理边际变化
        epu_smooth = epu.rolling(window=5).mean()
        
        # 计算 EPU 的 Z-score
        epu_mean = epu_smooth.rolling(window=self.zscore_window).mean()
        epu_std = epu_smooth.rolling(window=self.zscore_window).std()
        epu_z = (epu_smooth - epu_mean) / (epu_std + 1e-8)

        # 计算 VIX 的 Z-score
        vix_mean = vix.rolling(window=self.zscore_window).mean()
        vix_std = vix.rolling(window=self.zscore_window).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)

        # 构建政策-期权波动率共振冲击指标 (Unstructured + Options)
        composite_shock = vix_z + epu_z
        shock_mean = composite_shock.rolling(window=self.zscore_window).mean()
        shock_std = composite_shock.rolling(window=self.zscore_window).std()
        shock_z = (composite_shock - shock_mean) / (shock_std + 1e-8)

        # 二阶导数条件: 极值 + 衰竭
        
        # 1. 恐慌极端衰竭 -> 做多美债条件 (避险资金重新回流长债)
        # 条件A: 共振恐慌指数 Z-Score 处于极值 ( > 2.5 )
        # 条件B: 期权恐慌开始实质性衰竭 (VIX 下穿 3日均值 且 当日回落)
        vix_exhaustion = (vix < vix.rolling(window=self.exhaust_window).mean()) & (vix.diff() < 0)
        long_condition = (shock_z > self.trigger_threshold) & vix_exhaustion

        # 2. 自满极端反弹 -> 做空美债条件 (市场可能遭遇紧缩突袭)
        # 条件A: 宏观与期权呈现极度自满与确定性 ( Z-Score < -2.5 )
        # 条件B: 隐含波动率异动抬头 (VIX 上穿 3日均值 且 当日走高)
        vix_reversal = (vix > vix.rolling(window=self.exhaust_window).mean()) & (vix.diff() > 0)
        short_condition = (shock_z < -self.trigger_threshold) & vix_reversal

        # 生成信号 (脉冲型)
        signal[long_condition] = 1.0
        signal[short_condition] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, threshold={self.trigger_threshold})"