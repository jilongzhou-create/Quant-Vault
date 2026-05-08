import numpy as np
import pandas as pd

class UnstructuredEpuMomentumExhaustionFactor:
    """经济政策不确定性动量衰竭脉冲因子 (unstructured/unstructured)

    逻辑: 经济政策不确定性(EPU)基于新闻文本分析，反映宏观避险情绪。当不确定性短期内极端飙升并开始见顶回落时，标志着恐慌情绪峰值已过，资金确定性流入美债的避险趋势确立，产生看多脉冲；反之极度消退见底时产生看空脉冲。
    数据: usepuindxd (美国经济政策不确定性新闻指数)
    触发: EPU平滑后10日变动量的 252日 Z-Score > 2.5 且向下跌破3日均线(动能衰竭) -> 看多 (+1.0)
    输出: +1.0 看多美债, -1.0 看空美债，常态严格输出0.0
    """

    def __init__(self):
        self.name = 'unstructured_epu_momentum_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (Sniper Pulse) 初始信号必须全为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 禁止直接用原始水位，必须使用变化量。此处取5日均线以过滤新闻日度噪音，并计算10日动量
        epu_smooth = epu.rolling(window=5, min_periods=1).mean()
        epu_mom = epu_smooth.diff(10)
        
        # 计算 252 日滚动 Z-Score (反映相比过去一年的极端突变)
        mom_mean = epu_mom.rolling(window=252, min_periods=60).mean()
        mom_std = epu_mom.rolling(window=252, min_periods=60).std()
        mom_std = mom_std.replace(0, np.nan)  # 防除零
        
        epu_mom_z = (epu_mom - mom_mean) / mom_std
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 计算短期回落基准，必须等极值开始反转才交易
        mom_z_ma3 = epu_mom_z.rolling(window=3, min_periods=1).mean()
        
        # 条件1 & 条件2同时满足
        # 极端恐慌飙升并开始回落 (避险资金右侧确认流入美债)
        long_cond = (epu_mom_z > 2.5) & (epu_mom_z < mom_z_ma3)
        
        # 极端乐观极度发酵并见顶修复 (风险偏好极度膨胀后退潮，做空美债)
        short_cond = (epu_mom_z < -2.5) & (epu_mom_z > mom_z_ma3)
        
        # 输出脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"