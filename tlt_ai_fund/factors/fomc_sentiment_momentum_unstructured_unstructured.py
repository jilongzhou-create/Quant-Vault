import numpy as np
import pandas as pd

class FomcSentimentMomentumFactor:
    """FOMC 鹰鸽情绪动量突变因子 (unstructured/unstructured)

    逻辑: 绝对禁止直接使用阶梯状 FOMC 情绪的绝对值！本因子通过计算短期(10日)与长期(60日)情绪指数 EMA 的差值(MACD)，提取市场对美联储政策预期的"边际变化动量"。当该动量达到年度极端水平(Z-Score > 2.5，意味市场正处于鸽/鹰派预期骤变的狂热发酵期)，并且动量开始反转衰竭(当前低于3日均值)时，精准输出脉冲信号。这完美避免了在极速杀跌/暴涨阶段"接飞刀"，仅在定价冲击完成初始吸收的拐点介入趋势。
    数据: fomc_sentiment
    触发: 情绪动量的 252日 Z-Score > 2.5 且开始见顶衰退 (鸽派转向确立 → +1.0)；或 Z-Score < -2.5 且开始见底反升 (鹰派转向确立 → -1.0)。
    输出: 典型的狙击手级防飞刀脉冲，常态全为 0.0，仅在极端拐点跳变 +1.0 / -1.0。
    """

    def __init__(self, short_span: int = 10, long_span: int = 60, z_window: int = 252, z_threshold: float = 2.5):
        self.name = 'unstructured_fomc_sentiment_momentum_pulse'
        self.short_span = short_span
        self.long_span = long_span
        self.z_window = z_window
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，默认全局输出 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 容错处理
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # 前向填充会议日遗留的阶梯情绪得分
        fomc = data['fomc_sentiment'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 将低频阶梯转化为边际动量 (类似 MACD 捕获其动态斜率)
        ema_short = fomc.ewm(span=self.short_span, adjust=False).mean()
        ema_long = fomc.ewm(span=self.long_span, adjust=False).mean()
        momentum = ema_short - ema_long
        
        # 计算边际动量的极端程度 (滚动 1 年 Z-Score)
        mom_mean = momentum.rolling(window=self.z_window, min_periods=self.long_span).mean()
        mom_std = momentum.rolling(window=self.z_window, min_periods=self.long_span).std()
        
        # 加上微小 epsilon 防止在极度风平浪静年份的除零错误
        mom_z = (momentum - mom_mean) / (mom_std + 1e-8)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 计算近期平均动能，判断极端情绪是否已经停止加速并开始衰退
        mom_ma3 = momentum.rolling(window=3).mean()
        
        # 多头脉冲：极度鸽派突变发酵至顶点并开始衰竭 (动量极大但今日开始转弱)
        is_extreme_bull = mom_z > self.z_threshold
        is_exhausted_bull = momentum < mom_ma3
        
        # 空头脉冲：极度鹰派突变发酵至极点并开始回弹 (负动量极大但今日开始转强)
        is_extreme_bear = mom_z < -self.z_threshold
        is_exhausted_bear = momentum > mom_ma3
        
        # 获取潜在触发点
        bull_condition = is_extreme_bull & is_exhausted_bull
        bear_condition = is_extreme_bear & is_exhausted_bear
        
        # 狙击手强化滤网: 仅在从"未触发"转入"触发"的瞬间输出脉冲信号，消灭连续触发日
        bull_pulse = bull_condition & (~bull_condition.shift(1).fillna(False))
        bear_pulse = bear_condition & (~bear_condition.shift(1).fillna(False))
        
        # 输出最终信号
        signal[bull_pulse] = 1.0
        signal[bear_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(short_span={self.short_span}, long_span={self.long_span}, z_window={self.z_window}, z_threshold={self.z_threshold})"