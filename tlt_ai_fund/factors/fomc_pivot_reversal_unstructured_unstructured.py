import numpy as np
import pandas as pd

class FomcPivotReversalFactor:
    """FOMC情绪突变反转因子 (unstructured/unstructured)

    逻辑: 捕捉美联储政策预期的极端耗竭与边际反转。当FOMC情绪得分在过去一年达到极端高位(极度鸽派)或低位(极度鹰派)时，市场对于该方向的预期已极度拥挤。此时一旦最新声明发生逆向的边际变动(阶梯值跳跃反转)，即标志着拐点瞬间，触发单日脉冲捕捉超预期Price-in导致的TLT暴涨或暴跌。
    数据: fomc_sentiment (基于央行文本分析的鹰鸽情绪得分, 阶梯状)
    触发: 前一日的 252日 Z-Score 处于极端状态 (> 2.5 或 < -2.5)，且当日出现反向的边际跳跃 (diff > 0 鹰转鸽, diff < 0 鸽转鹰)。
    输出: 仅在反转当天产生脉冲信号 (+1.0/-1.0)，其余所有交易日休眠返回 0.0。
    """

    def __init__(self, zscore_window: int = 252, zscore_threshold: float = 2.5):
        self.name = 'fomc_pivot_reversal'
        self.zscore_window = zscore_window
        self.zscore_threshold = zscore_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查是否包含所需因子数据，避免跨域并妥善处理缺失
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        sentiment = data['fomc_sentiment']
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # fomc_sentiment是每年约8次变动的阶梯数据，必须使用.diff()捕捉预期改变的瞬间
        marginal_change = sentiment.diff(1)
        
        # 计算情绪水位的长期极端程度
        # 采用252个交易日(约1年，覆盖约8次FOMC会议周期)进行基准评估
        rolling_mean = sentiment.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).mean()
        rolling_std = sentiment.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).std()
        
        # 避免除以0的情况
        rolling_std = rolling_std.replace(0.0, np.nan)
        zscore = (sentiment - rolling_mean) / rolling_std
        
        # 使用昨日的Z-Score来判断极值，确保在发生边际跳跃之前，市场预期处于极端拥挤的耗竭状态
        prev_zscore = zscore.shift(1)
        
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 初始化全0 Series，确保非触发日完全休眠
        signal = pd.Series(0.0, index=data.index)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 只有在预期耗竭 (极端高/低位) 且 开始反转 (边际跳跃符号相反) 的瞬间同时满足才触发
        
        # 极端鹰派耗竭 (Z-Score < -2.5) + 意外转鸽脉冲 (diff > 0) -> 利率暴跌，看多TLT
        long_condition = (prev_zscore < -self.zscore_threshold) & (marginal_change > 0)
        
        # 极端鸽派耗竭 (Z-Score > 2.5) + 意外转鹰脉冲 (diff < 0) -> 利率飙升，看空TLT
        short_condition = (prev_zscore > self.zscore_threshold) & (marginal_change < 0)
        
        signal.loc[long_condition] = 1.0
        signal.loc[short_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, zscore_threshold={self.zscore_threshold})"