import numpy as np
import pandas as pd

class VixExhaustionPulseFactor:
    """VIX恐慌衰竭与爆发脉冲 (unstructured/options)

    逻辑: 捕捉期权隐含波动率(VIX)的微观结构突变。当VIX单日暴涨(边际变化Z-Score>2.5)且确立涨势时，避险情绪爆发，资金涌入美债，产生看多脉冲；当VIX处于极端高位(水位Z-Score>2.5)且动量衰竭开始回落时，避险情绪消退，资金流出美债，产生看空脉冲。
    数据: vixcls
    触发: 多头(避险爆发): VIX单日变化Z-Score > 2.5 且 VIX > 3日均值；空头(恐慌衰竭): VIX绝对水位Z-Score > 2.5 且 VIX < 3日均值 且 VIX单日回落
    输出: +1.0 表示避险爆发看多TLT，-1.0 表示恐慌消退看空TLT，其余为0.0脉冲休眠
    """

    def __init__(self, window: int = 252, z_threshold: float = 2.5):
        self.name = 'vix_exhaustion_pulse'
        self.window = window           # 252个交易日(约1年)作为长周期基准
        self.z_threshold = z_threshold # 2.5倍标准差代表极端尾部事件

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 缺失数据处理
        if 'vixcls' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        vix = data['vixcls'].ffill()
        signal = pd.Series(0.0, index=vix.index, name=self.name)

        # 铁律3: 边际变化 (获取一阶导数捕捉突变瞬间)
        vix_diff = vix.diff(1)
        
        # 长周期水位特征
        vix_mean = vix.rolling(self.window).mean()
        vix_std = vix.rolling(self.window).std()
        vix_z = (vix - vix_mean) / vix_std

        # 长周期边际变化特征
        vix_diff_mean = vix_diff.rolling(self.window).mean()
        vix_diff_std = vix_diff.rolling(self.window).std()
        vix_diff_z = (vix_diff - vix_diff_mean) / vix_diff_std

        # 短期动量基准 (用于二阶导数衰竭确认)
        vix_ma3 = vix.rolling(3).mean()

        # 多头触发 (Risk-Off 避险情绪骤升买入美债)
        # 条件1: 波动率单日跳跃达到极端水平 (Z > 2.5)
        # 条件2: 趋势得到确认 (高于3日均线，剔除日内冲高回落的假突破)
        long_cond = (vix_diff_z > self.z_threshold) & (vix > vix_ma3)

        # 空头触发 (Risk-On 恐慌情绪消退做空美债)
        # 铁律2: 二阶导数 (绝对禁止 VIX > 极值直接做空！)
        # 条件1: 波动率绝对水位处于极度恐慌高位 (Z > 2.5)
        # 条件2: 恐慌开始衰竭 (单日开始回落 且 跌破3日均线动量反转)
        short_cond = (vix_z > self.z_threshold) & (vix_diff < 0) & (vix < vix_ma3)

        # 铁律1: 零值休眠 (Sniper Pulse)
        # 常态下保持0.0，仅在极端脉冲触发时赋值
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold})"