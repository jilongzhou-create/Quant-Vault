import numpy as np
import pandas as pd

class EpuPanicExhaustionFactor:
    """政策不确定性恐慌见顶回归因子 (panic_mean_reversion/unstructured)

    逻辑: 极高的经济政策不确定性(EPU)往往对应美股短期的情绪极致恐慌与见底。
          当基于新闻文本的EPU指数(usepuindxd)处于过去一年的极高位(Z-Score > 1.5), 
          且近期动量由正转负(恐慌不再恶化且见顶回落)时, 产生强烈的抄底买入信号。
          相反, 当EPU在相对平稳期突然向上暴增时, 提示宏观黑天鹅恶化初期, 尚未产生极值衰竭, 输出短线看空。
    数据: [usepuindxd] (US Economic Policy Uncertainty Index)
    输出: 1.0表示政策不确定性见顶衰竭(强烈看多), -1.0表示不确定性突发飙升初期(看空趋势恶化)
    触发条件: EPU的Z-Score > 1.5 且今日及3日动量 < 0 时输出 +1.0; 在均值以下突然爆发3日暴增>1.5倍标准差时输出 -1.0。预期Trigger Rate 5-10%
    """

    def __init__(self, window=252, z_threshold=1.5, momentum_window=3):
        self.name = 'epu_panic_exhaustion_pulse'
        self.window = window
        self.z_threshold = z_threshold
        self.momentum_window = momentum_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 如果缺少所需数据列，则直接返回全0的休眠信号
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        epu = data['usepuindxd'].ffill()
        
        # 计算 252 日滚动统计量，以定义长期基准环境
        roll_mean = epu.rolling(self.window).mean()
        roll_std = epu.rolling(self.window).std()
        
        # 避免标准差为0导致的除零异常
        roll_std = roll_std.replace(0.0, np.nan)
        z_score = (epu - roll_mean) / roll_std
        
        # 计算动量变化 (捕获预期边际反转，坚决禁止仅依据绝对值输出)
        diff_1 = epu.diff(1)
        diff_3 = epu.diff(self.momentum_window)
        
        # 初始化零值休眠信号
        signal = pd.Series(0.0, index=epu.index)
        
        # 多头信号 (恐慌衰竭抄底)：
        # 条件1: 不确定性极高 (Z-Score > 1.5, 处于极端恐慌状态)
        # 条件2: 今日边际回落 (diff_1 < 0, 防接飞刀，二阶导数开始向下)
        # 条件3: 3日内明确脱离顶部 (diff_3 < 0)
        long_cond = (z_score > self.z_threshold) & (diff_1 < 0) & (diff_3 < 0)
        
        # 空头信号 (黑天鹅事件爆发初期顺势做空)：
        # 条件1: 之前属于相对正常时期 (3天前的 Z-Score < 0.5)
        # 条件2: 短期内突发惊恐暴增 (近3日涨幅超过 1.5 倍的长期波动率)
        # 条件3: 今日仍在恶化发酵中 (diff_1 > 0)
        short_cond = (z_score.shift(self.momentum_window) < 0.5) & (diff_3 > roll_std * 1.5) & (diff_1 > 0)
        
        # 组装脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        # 填补NaN (初始化期)
        signal = signal.fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold})"