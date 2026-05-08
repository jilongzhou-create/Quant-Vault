import numpy as np
import pandas as pd

class UnstructuredVolatilityReversalFactor:
    """波动率极值与非结构化政策不确定性拥挤反转因子 (volatility/unstructured)

    逻辑: 监控基于新闻文本的经济政策不确定性(EPU)与跨资产恐慌(VIX)的共振飙升。绝对禁止直接买入极端高位，只有当政策不确定性或VIX飙升至极端水平，且两者同时向下拐头时，才意味着抛售衰竭和流动性预期突变，此时输出买入脉冲。
    数据: usepuindxd (经济政策不确定性指数，非结构化文本数据), vixcls (VIX波动率)
    触发: 至少一者的 252日 Z-Score > 2.5 (极值) 且另一者 > 1.0 (跨域确认)，并严格配合两者的日度 diff() < 0 且低于3日均值 (衰竭拐点确认)。
    输出: +1.0 看多美债 (恐慌抛售见顶衰竭，宽松预期发酵), -1.0 看空美债 (过度贪婪见底并开始反弹), 常态 0.0
    """

    def __init__(self):
        self.name = 'unstructured_volatility_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 检查必要数据列是否存在
        required_cols = ['usepuindxd', 'vixcls']
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)
        
        # 2. 数据预处理 (前向填充处理周末及假日缺失)
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        
        # 3. 计算 252日滚动 Z-Score 以捕捉宏观周期内的极端异动
        window = 252
        epu_mean = epu.rolling(window=window).mean()
        epu_std = epu.rolling(window=window).std()
        epu_z = (epu - epu_mean) / epu_std.replace(0, np.nan)
        
        vix_mean = vix.rolling(window=window).mean()
        vix_std = vix.rolling(window=window).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        # 4. 二阶导数衰竭条件 (边缘变化)
        # 必须开始回落: 今日水位差分 < 0 且 今日水位绝对值跌穿短波 3日均值
        epu_exhaust = (epu.diff() < 0) & (epu < epu.rolling(window=3).mean())
        vix_exhaust = (vix.diff() < 0) & (vix < vix.rolling(window=3).mean())
        
        # 反向衰竭条件: 今日开始反弹 (太平盛世被打破)
        epu_rebound = (epu.diff() > 0) & (epu > epu.rolling(window=3).mean())
        vix_rebound = (vix.diff() > 0) & (vix > vix.rolling(window=3).mean())
        
        # 5. 脉冲触发条件
        # 多头触发 (恐慌见顶回落): 至少一个极度狂飙 (Z>2.5), 且另一个给出偏高位确认 (Z>1.0)
        panic_spike = ((epu_z > 2.5) & (vix_z > 1.0)) | ((vix_z > 2.5) & (epu_z > 1.0))
        # 二阶导数必须同时满足：极值 + 双双回落衰竭 = 反转买点
        long_trigger = panic_spike & epu_exhaust & vix_exhaust
        
        # 空头触发 (极度乐观见底反转): 至少一个极度低迷 (Z<-2.0), 且另一个给出偏低位确认 (Z<-1.0)
        greed_spike = ((epu_z < -2.0) & (vix_z < -1.0)) | ((vix_z < -2.0) & (epu_z < -1.0))
        # 二阶导数同时满足：绝对低谷 + 突发反弹 = 风险重估卖点
        short_trigger = greed_spike & epu_rebound & vix_rebound
        
        # 6. 构造脉冲信号 (铁律1: 零值休眠)
        signal = pd.Series(0.0, index=data.index)
        signal.loc[long_trigger] = 1.0
        signal.loc[short_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"