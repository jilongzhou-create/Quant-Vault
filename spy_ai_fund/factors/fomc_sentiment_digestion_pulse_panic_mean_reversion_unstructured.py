import numpy as np
import pandas as pd

class EpuPanicExhaustionFactor:
    """Policy Uncertainty Panic Exhaustion Factor (panic_mean_reversion/unstructured)

    逻辑: EPU(经济政策不确定性指数)基于海量非结构化新闻提炼。当不确定性刚上穿中枢升温时(轻度恐慌发酵)，
          市场处于钝刀割肉的恶化趋势；而当不确定性飙升至极度悲观的历史极值(Z>1.5)且确认见顶回落的第一时间(恐慌衰竭)，
          往往是美股独有的“V型反转”绝佳抄底时刻。
    数据: usepuindxd (美国经济政策不确定性指数)
    输出: +1.0 (恐慌极值且开始回落, 狙击式抄底); -1.0 (不确定性突破升温, 规避下跌)
    触发条件: 
        买入: 252日Z-Score>1.5，且其5日均线昨日上升今日下降(出现极值向下拐点)，脉冲持续最多5天。
        卖出: 252日Z-Score由下上穿0.5(情绪恶化)，脉冲持续最多5天。
        目标 Trigger Rate 完美控制在 5% - 15% 的狙击手范围内。
    """

    def __init__(self):
        self.name = 'epu_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 1. 基础数据处理: 前向填充处理非交易日，使用5日移动平均过滤EPU极其剧烈的日度噪音
        epu = data['usepuindxd'].ffill()
        epu_smooth = epu.rolling(window=5, min_periods=1).mean()
        
        # 2. 宏观水温测量: 计算252日(约一年)滚动Z-Score，严防未来数据
        roll_mean = epu_smooth.rolling(window=252, min_periods=63).mean()
        roll_std = epu_smooth.rolling(window=252, min_periods=63).std()
        
        # 防呆处理: std为0时替换为NaN避免除零报警
        roll_std = roll_std.replace(0.0, np.nan)
        epu_z = (epu_smooth - roll_mean) / roll_std
        
        # 3. 计算边际动能 (铁律：二阶导数识别转折点)
        epu_diff = epu_smooth.diff()
        prev_diff = epu_smooth.shift(1).diff()
        
        # 4. 捕捉极度恐慌见顶瞬间 (买入逻辑)
        # 条件: Z-Score > 1.5 (极度恐慌) 且 昨天还在升高今日转跌 (拐点确立)
        is_peak = (epu_z > 1.5) & (epu_diff < 0.0) & (prev_diff >= 0.0)
        
        # 脉冲维持: 见顶后的5个自然日内，只要恐慌依然在退潮(diff < 0)，就维持买入信号
        buy_pulse = is_peak.rolling(window=5, min_periods=1).max() == 1.0
        buy_cond = buy_pulse & (epu_diff < 0.0)
        
        # 5. 捕捉钝刀割肉恶化瞬间 (卖出逻辑)
        # 条件: Z-Score刚刚上穿 0.5 (不确定性开始显著高于年内均值，轻度恐慌发酵)
        is_breakout = (epu_z > 0.5) & (epu_z.shift(1) <= 0.5)
        
        # 脉冲维持: 突破后的5个自然日内，只要恐慌依然在加剧(diff > 0)，就维持卖出信号
        sell_pulse = is_breakout.rolling(window=5, min_periods=1).max() == 1.0
        sell_cond = sell_pulse & (epu_diff > 0.0)
        
        # 6. 整合休眠与爆发信号 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"