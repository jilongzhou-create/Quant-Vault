import numpy as np
import pandas as pd

class UnstructuredFomcSentimentPulseFactor:
    """FOMC情绪突变与曲线共振脉冲因子 (unstructured/unstructured)

    逻辑: 捕捉美联储FOMC声明情绪的边际极值突变(NLP文本情绪阶梯数据)。单纯的会议声明转鸽可能遭遇市场反向交易, 必须看到短端利率(dgs2)切实回落且收益率曲线(t10y2y)发生牛市变陡(Bull Steepening)共振, 才能确认市场已 Price-in 降息预期, 从而触发安全的高胜率做多脉冲。反之亦然。完美规避接飞刀。
    数据: fomc_sentiment (NLP情绪分), dgs2 (短端政策利率), t10y2y (期限利差)
    触发: fomc_sentiment 5日差值的 252日 Z-Score > 2.5 (情绪极值) + dgs2 3日下行 (衰竭确认) + t10y2y 5日差值 > 0 (牛陡确认)
    输出: 仅在产生共振的短期窗口内输出 +1.0 (看多) 或 -1.0 (看空), 常态绝对为 0.0, 典型的狙击手级脉冲。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_sentiment_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化常态休眠信号
        signal = pd.Series(0.0, index=data.index)

        # 检查依赖的核心字段是否齐全
        required_cols = ['fomc_sentiment', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal

        # ---------------------------------------------------------------------
        # 铁律3: 边际变化 (Marginal Change)
        # fomc_sentiment 是每年约8次变动的阶梯低频数据, 严禁使用绝对值!
        # 使用 5 日差值将突变瞬间转化为脉冲特征，使得突变后的极短几天内具有高动能
        # ---------------------------------------------------------------------
        sentiment_diff = data['fomc_sentiment'].diff(5)

        # 动态滚动计算 Z-Score (使用 252 个交易日约1年的窗口，捕捉年内政策周期的相对突变)
        roll_mean = sentiment_diff.rolling(window=252, min_periods=63).mean()
        roll_std = sentiment_diff.rolling(window=252, min_periods=63).std()
        
        # 避免除以 0 的极小概率事件
        roll_std = roll_std.replace(0.0, np.nan)
        z_score = (sentiment_diff - roll_mean) / roll_std

        # ---------------------------------------------------------------------
        # 铁律2: 二阶导数与衰竭 (Anti-Catch-Falling-Knife)
        # 避免指标极值时逆势硬接飞刀。结合对政策最敏感的 dgs2 及 t10y2y 进行衰竭确认。
        # 只有在政策情绪发酵的同时, 实体收益率市场发生跟随(即衰竭), 才予以入场触发。
        # ---------------------------------------------------------------------
        dgs2_diff3 = data['dgs2'].diff(3)
        t10y2y_diff5 = data['t10y2y'].diff(5)

        # 做多共振脉冲 (鸽派突变 + 短端下行确认 + 曲线牛市变陡)
        long_cond = (
            (z_score > 2.5) &              # 极度鸽派边际突变
            (dgs2_diff3 < 0.0) &           # 短端利率回落(反接飞刀，确认市场交易降息)
            (t10y2y_diff5 > 0.0)           # 曲线变陡确认
        )

        # 做空共振脉冲 (鹰派突变 + 短端上行确认 + 曲线熊市变平/倒挂)
        short_cond = (
            (z_score < -2.5) &             # 极度鹰派边际突变
            (dgs2_diff3 > 0.0) &           # 短端利率上行(反接飞刀，确认市场交易加息)
            (t10y2y_diff5 < 0.0)           # 曲线变平/倒挂加深确认
        )

        # ---------------------------------------------------------------------
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 非触发日必然保持为 0.0
        # ---------------------------------------------------------------------
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        # 处理可能产生的 NaN, 命名并返回
        signal = signal.fillna(0.0)
        signal.name = self.name

        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"