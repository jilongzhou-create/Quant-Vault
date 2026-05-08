import numpy as np
import pandas as pd

class PolicyUncertaintyShockFactor:
    """政策不确定性脉冲因子 (unstructured/options)

    逻辑: 捕捉由非结构化新闻文本驱动的经济政策不确定性(EPU)的极端爆发与衰竭。当不确定性极端飙升且动能见顶回落时，标志着突发避险情绪已被完全定价，伴随政策干预预期推动长端美债反弹；当不确定性断崖式消散且衰竭时，风险偏好回归导致资金从债市撤出。严格属于脉冲型驱动而非连续状态。
    数据: usepuindxd (经济政策不确定性指数)
    触发: EPU 5日边际增量的 Z-Score > 2.5 (或 < -2.5), 且满足二阶导数衰竭条件(与3日均线死叉/金叉)
    输出: +1.0 (恐慌极值衰竭买入), -1.0 (利空彻底落地抛售), 0.0 (常态休眠)
    """

    def __init__(self):
        self.name = 'policy_uncertainty_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化信号, 严格遵守铁律1: 常态下必须休眠为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        epu = data['usepuindxd']
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 禁止使用绝对水位, 必须使用边际变化量捕捉预期突变的瞬间
        epu_diff = epu.diff(5)
        
        # 计算动量的 252 日滚动 Z-Score 以评估极端事件
        roll_mean = epu_diff.rolling(window=252).mean()
        roll_std = epu_diff.rolling(window=252).std()
        
        # 防止除零警告
        roll_std = roll_std.replace(0, np.nan)
        zscore = (epu_diff - roll_mean) / roll_std
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 绝对禁止在恐慌主升浪中接飞刀, 必须等待动能衰竭
        
        # 多头条件：政策恐慌极端爆发 (Z > 2.5) 且爆发动能开始衰竭 (动量回落到3日均线下方)
        # 经济学含义: 避险情绪达到高潮开始退潮, 市场开始 Price-in 政策干预/降息兜底 -> 做多 TLT
        cond_long_extreme = zscore > 2.5
        cond_long_exhaustion = epu_diff < epu_diff.rolling(window=3).mean()
        
        # 空头条件：政策恐慌极端消退 (Z < -2.5) 且消退动能开始衰竭 (动量回升到3日均线上方)
        # 经济学含义: 黑天鹅风险彻底解除 (如债务上限法案突然通过), 风险偏好狂热, 资金流出债市 -> 做空 TLT
        cond_short_extreme = zscore < -2.5
        cond_short_exhaustion = epu_diff > epu_diff.rolling(window=3).mean()
        
        # 赋值脉冲信号
        signal.loc[cond_long_extreme & cond_long_exhaustion] = 1.0
        signal.loc[cond_short_extreme & cond_short_exhaustion] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"