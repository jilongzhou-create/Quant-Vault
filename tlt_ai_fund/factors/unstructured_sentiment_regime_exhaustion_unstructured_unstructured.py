import numpy as np
import pandas as pd

class UnstructuredSentimentRegimeExhaustionFactor:
    """Unstructured FOMC Sentiment Regime Exhaustion (unstructured/unstructured)

    逻辑: 捕捉美联储中长期鹰鸽情绪极值后的边际衰竭与反转。FOMC情绪得分是低频阶梯数据，通过计算长短EMA的差值(MACD)，
          将其转化为连续的“政策情绪动量”。当长周期的政策情绪动量达到极值(严重超买/超卖)并且动量的二阶导数发生反转(开始衰竭)时，
          意味着政策冲击已被市场完全Price-in并开始消散，此时触发国债的反向脉冲，完美避免了在动量最强时接飞刀。
    数据: fomc_sentiment (基于LLM的FOMC声明鹰鸽文本情感得分)
    触发: 情绪MACD(63, 252)的 2年 Z-Score 绝对值 > 1.5 (约对应10%的历史极端区间) + MACD三日变化量发生反转 (衰竭条件)
    输出: 脉冲型信号。极端鹰派动量开始衰竭回升时输出+1.0 (做多TLT)，极端鸽派动量衰竭回落时输出-1.0 (做空TLT)。常态下为0.0。
    """

    def __init__(self):
        self.name = 'unstructured_sentiment_regime_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 基础数据校验，缺失则返回全0
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 1. 基础数据处理，前向填充以维持Step Function状态
        sentiment = data['fomc_sentiment'].ffill().fillna(0.0)

        # 2. 计算宏观情绪动量 (边际变化铁律)
        # 绝对禁止使用原值。使用EMA将低频阶梯数据转化为平滑的连续动量。
        # EMA(63)代表一季度中期情绪边际，EMA(252)代表一年期长期情绪底座
        ema_short = sentiment.ewm(span=63, adjust=False).mean()
        ema_long = sentiment.ewm(span=252, adjust=False).mean()
        macd_mom = ema_short - ema_long

        # 3. 计算动量的极端程度 (Z-Score)
        # 使用两年的滚动窗口来评估动量的相对极端性，适应不同宏观周期的波动率
        rolling_mean = macd_mom.rolling(window=504, min_periods=252).mean()
        rolling_std = macd_mom.rolling(window=504, min_periods=252).std()
        
        # 计算Z-Score，加入微小常数防止除以0
        zscore = (macd_mom - rolling_mean) / (rolling_std + 1e-8)

        # 4. 衰竭与反转条件 (二阶导数铁律: Anti-Catch-Falling-Knife)
        # 计算动量的短期边际变化。只有当旧趋势的动量停止加速并开始回落时，才是安全的抄底/逃顶时机
        mom_change = macd_mom.diff(3)

        # 5. 生成狙击手级脉冲信号 (零值休眠铁律)
        # 初始化为0.0，只在特定窗口期触发脉冲，目标Trigger Rate约5%-10%
        signal = pd.Series(0.0, index=data.index)

        # 多头触发: 处于极端鹰派状态 (zscore < -1.5) 且 鹰派动量开始衰竭回升 (mom_change > 0)
        # 此时市场已经被极度鹰派预期压测，边际转暖即刻引发美债报复性反弹
        long_cond = (zscore < -1.5) & (mom_change > 0)
        
        # 空头触发: 处于极端鸽派状态 (zscore > 1.5) 且 鸽派动量开始衰竭下降 (mom_change < 0)
        # 此时市场已将降息预期打满，利好出尽，动量衰竭即刻引发长端美债回调
        short_cond = (zscore > 1.5) & (mom_change < 0)

        # 赋值触发信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(macd_short=63, macd_long=252, z_window=504, z_threshold=1.5)"