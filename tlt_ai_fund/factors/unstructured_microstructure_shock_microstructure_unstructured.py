import numpy as np
import pandas as pd

class UnstructuredEpuExhaustionFactor:
    """经济政策不确定性新闻NLP衰竭因子 (microstructure/unstructured)

    逻辑: 每日新闻隐含的经济政策不确定性(EPU)作为非结构化NLP数据的代表，其飙升常伴随市场微观流动性枯竭。当恐慌见顶(Z>1.2)并开始衰竭回落时，流动性危机解除，避险资金回流美债；反之极度贪婪见底且风险抬头时，抛售美债。符合零值休眠和二阶导数反飞刀铁律。
    数据: usepuindxd (每日新闻经济政策不确定性指数)
    触发: 252日 Z-Score > 1.2 且 3日均值 < 10日均值 (恐慌衰竭脉冲)；Z-Score < -1.2 且 3日均值 > 10日均值 (极度贪婪反转脉冲)
    输出: 脉冲型信号，触发时输出 +1.0 看多 或 -1.0 看空，非触发日严格输出 0.0
    """

    def __init__(self):
        self.name = 'unstructured_epu_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 的信号序列，严格遵守零值休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖列是否存在，避免报错
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 获取EPU数据并前向填充处理缺失值
        epu = data['usepuindxd'].ffill()
        
        # 计算短期和中期平滑均值，用于捕获动量反转和边际变化 (边际变化铁律)
        epu_ma3 = epu.rolling(window=3).mean()
        epu_ma10 = epu.rolling(window=10).mean()
        
        # 计算长周期的滚动 Z-Score 来识别极值水位
        epu_mean252 = epu_ma3.rolling(window=252).mean()
        epu_std252 = epu_ma3.rolling(window=252).std()
        
        epu_zscore = (epu_ma3 - epu_mean252) / (epu_std252 + 1e-8)
        
        # --- 极值触发条件 + 衰竭反转条件 (遵守 Anti-Catch-Falling-Knife 二阶导数铁律) ---
        # 适度放宽 Z-Score 阈值至 1.2，确保触发率落在 5% 到 15% 之间
        
        # 条件1 (看多美债): 新闻不确定性极度高涨 (Z > 1.2) 且 边际动量开始衰竭回落 (MA3 < MA10)
        # 意味着恐慌已经见顶，抛售潮结束，避险配置资金开始重返美债
        long_cond = (epu_zscore > 1.2) & (epu_ma3 < epu_ma10)
        
        # 条件2 (看空美债): 新闻不确定性极度低迷 (Z < -1.2) 且 边际动量开始抬头回升 (MA3 > MA10)
        # 意味着市场从极度自满中苏醒，开始重新定价通胀或紧缩风险，引发美债抛售
        short_cond = (epu_zscore < -1.2) & (epu_ma3 > epu_ma10)
        
        # 赋值脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"