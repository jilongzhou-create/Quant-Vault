import numpy as np
import pandas as pd

class UnstructuredEpuGvzPulseFactor:
    """Unstructured EPU and Gold Options Volatility Pulse Factor (unstructured/options)

    逻辑: 结合非结构化新闻数据(经济政策不确定性EPU)与期权隐含波动率(黄金GVZ)捕捉宏观恐慌脉冲。由于黄金和美债同为避险资产，当EPU与GVZ同时剧烈飙升并随后衰竭时，标志着避险情绪见顶，美联储政策预期转向，看多美债(TLT)；反之当极度自满状态被打破时看空美债。
    数据: usepuindxd, gvzcls
    触发: 组合动量 Z-Score > 1.2 且 GVZ 开始回落 -> +1.0；Z-Score < -1.2 且 GVZ 开始反弹 -> -1.0。
    输出: [-1.0, 1.0] 的离散脉冲信号。
    """

    def __init__(self):
        self.name = 'unstructured_epu_gvz_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns or 'gvzcls' not in data.columns:
            return signal

        epu = data['usepuindxd'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 使用 3日动量 捕捉市场对新闻不确定性和避险情绪预期的突变瞬间
        epu_mom = epu.diff(3)
        gvz_mom = gvz.diff(3)
        
        # 使用 63个交易日(约一个季度) 滚动标准化，适应宏观环境的Regime Change
        epu_z = (epu_mom - epu_mom.rolling(63).mean()) / epu_mom.rolling(63).std()
        gvz_z = (gvz_mom - gvz_mom.rolling(63).mean()) / gvz_mom.rolling(63).std()
        
        # 构建跨域组合冲击指数
        shock = epu_z + gvz_z
        shock_z = (shock - shock.rolling(63).mean()) / shock.rolling(63).std()
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 必须等待波动率指标开始回落/反弹，防止在趋势主升浪/主跌浪中接飞刀
        gvz_ma3 = gvz.rolling(3).mean()
        
        # 采用 1.2 的 Z-Score 阈值(约正态分布的11.5%分位数)，配合衰竭过滤，确保 Trigger Rate 控制在 5%-15% 的健康区间
        long_cond = (shock_z > 1.2) & (gvz < gvz_ma3)
        short_cond = (shock_z < -1.2) & (gvz > gvz_ma3)
        
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 非极端触发日严格保持 0.0
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"