import numpy as np
import pandas as pd

class FomcSentimentCrowdingReversalFactor:
    """FOMC情绪拥挤反转脉冲因子 (volatility/unstructured)

    逻辑: 在加息/降息周期的末端，市场往往对美联储的单向政策预期产生"拥挤交易" (Crowding)。
          本因子利用非结构化 NLP 情绪得分，专门狙击政策预期极端拥挤后的边际突变。
          这是一个典型的脉冲因子，严格执行不接飞刀法则：只在央行确认边际转向(动量衰竭)的瞬间，
          输出为期 5 个交易日的脉冲信号（代表债市吸收政策冲击并进行大规模空头回补/多头平仓的时间窗口）。
    数据: fomc_sentiment (央行文本情绪得分)
    触发:
          条件1 (极端拥挤): 会议前一天的情绪得分 252日 Z-Score 偏离均值超 1.0 个标准差 (处于统计学上的单边拥挤区间)
          条件2 (衰竭反转): 最新声明发布导致情绪得分发生反向跳跃 (边际变化 diff 与拥挤极值方向相反)
    输出: 鹰转鸽突变输出 +1.0 (多头反转), 鸽转鹰突变输出 -1.0 (空头反转)。非窗口期信号严格为 0.0。
    """

    def __init__(self):
        self.name = 'fomc_sentiment_crowding_reversal'
        self.window = 252       # 252个交易日，代表1个完整宏观自然年的基准周期
        self.min_periods = 60   # 最少需要约1个季度的历史数据来建立基准分布
        self.z_threshold = 1.0  # 1.0个标准差，统计学上区分常态与极端单边区间的标准阈值
        self.pulse_days = 5     # 脉冲维持5个交易日，即1个自然周，符合机构资金调整头寸的典型吸收周期

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 异常与缺失列处理
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 数据前向填充，FOMC 数据为阶梯状低频跳跃数据
        fomc = data['fomc_sentiment'].ffill()

        # 计算年度维度的 Z-Score 绝对水位
        fomc_mean = fomc.rolling(window=self.window, min_periods=self.min_periods).mean()
        fomc_std = fomc.rolling(window=self.window, min_periods=self.min_periods).std()
        fomc_std = fomc_std.replace(0, np.nan) # 防止除零错误
        zscore = (fomc - fomc_mean) / fomc_std

        # 铁律3: 边际变化 (使用 .diff() 捕捉预期跳跃的瞬间)
        sentiment_diff = fomc.diff()

        # 铁律2: 二阶导数 (基于"突变前"的极值状态 + "突变日"的衰竭确认)
        # 提取昨日的 Z-Score 作为环境锚点
        prev_zscore = zscore.shift(1)

        # 触发器计算：
        # 做多脉冲: 前期极度鹰派(Z < -1.0) 且 最新会议释放鸽派转向信号(diff > 0)
        buy_trigger = (prev_zscore < -self.z_threshold) & (sentiment_diff > 0.0)

        # 做空脉冲: 前期极度鸽派(Z > 1.0) 且 最新会议释放鹰派转向信号(diff < 0)
        sell_trigger = (prev_zscore > self.z_threshold) & (sentiment_diff < 0.0)

        # 铁律1: 零值休眠 (Sniper Pulse)
        # 将 T 日的瞬间触发，通过 rolling max 延长至 5 个交易日的脉冲窗口
        # 满足 Trigger Rate 5%-15% 的目标，绝不输出连续因子
        buy_pulse = buy_trigger.rolling(window=self.pulse_days, min_periods=1).max() > 0
        sell_pulse = sell_trigger.rolling(window=self.pulse_days, min_periods=1).max() > 0

        # 初始化休眠信号
        signal = pd.Series(0.0, index=data.index)

        # 注入脉冲 (赋值机制保证了边界在 [-1.0, 1.0] 内)
        signal[buy_pulse] = 1.0
        signal[sell_pulse] = -1.0

        # 清除初期无数据阶段可能产生的 NaN
        signal = signal.fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold}, pulse_days={self.pulse_days})"