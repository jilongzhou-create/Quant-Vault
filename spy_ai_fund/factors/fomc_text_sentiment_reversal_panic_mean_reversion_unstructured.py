import numpy as np
import pandas as pd

class FomcTextSentimentReversalFactor:
    """FOMC非结构化文本情绪反转因子 (panic_mean_reversion/unstructured)

    逻辑: 捕捉美联储货币政策口风变化的边际瞬间。当极端鹰派情绪(宏观恐慌)边际软化, 或发生显著鸽派突变时, 恐慌情绪衰竭, 输出看多脉冲; 当情绪突然由宽转紧时输出看空脉冲。
    数据: fomc_sentiment (基于LLM解析的FOMC声明鹰鸽情感得分)
    输出: +1.0 表示紧缩恐慌衰竭并转向(看多), -1.0 表示突遭鹰派打击(趋势走坏看空)
    触发条件: 边际突变(diff>0.25或<-0.25)、零轴反转, 或极度恐慌衰竭(<-0.5且diff>=0.1)。脉冲在极短的后3天内延续, 预期Trigger Rate约 6%-10%。
    """

    def __init__(self):
        self.name = 'fomc_text_sentiment_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理缺失列，返回休眠零值
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        # FOMC数据为低频阶梯状, 必须先向后填充以获取当前日期的可用预期值
        fomc = data['fomc_sentiment'].ffill()
        
        # 边际变化铁律: 绝对禁止直接使用绝对值, 必须计算动量变化
        fomc_prev = fomc.shift(1)
        fomc_diff = fomc.diff()

        # --- 强烈看多 (+1.0) 抄底与衰竭逻辑 ---
        
        # 1. 鸽派大突变 (边际跳跃代表预期反转)
        dovish_surge = fomc_diff >= 0.25
        
        # 2. 零轴穿越 (情绪由看空反转为看多)
        turn_positive = (fomc_prev < 0.0) & (fomc > 0.0)
        
        # 3. 极端恐慌衰竭 (二阶导数铁律精髓)
        # 在极度鹰派(-0.5以下)的高压恐慌下, 只要态度出现微弱软化(>=0.1), 即可确认为"见顶回落"买点
        extreme_panic = fomc_prev <= -0.5
        panic_exhaustion = extreme_panic & (fomc_diff >= 0.1)

        buy_trigger = dovish_surge | turn_positive | panic_exhaustion

        # --- 趋势恶化 (-1.0) 看空逻辑 ---
        
        # 1. 鹰派大突变 (轻微恐慌或超预期打击导致趋势走坏)
        hawkish_surge = fomc_diff <= -0.25
        
        # 2. 零轴向下穿越 (情绪由看多转看空)
        turn_negative = (fomc_prev > 0.0) & (fomc < 0.0)
        
        sell_trigger = hawkish_surge | turn_negative

        # --- 脉冲信号组装 ---
        raw_signal = pd.Series(0.0, index=data.index)
        raw_signal[buy_trigger] = 1.0
        # 避免同时触发, 卖空逻辑后置降级处理
        raw_signal[sell_trigger & ~buy_trigger] = -1.0

        # 【零值休眠铁律】:
        # 每年仅约 8 次 FOMC 会议，如果在突变当天单日触发，Trigger Rate 不到 3%。
        # 为了保证 5%~15% 的目标触发率，且表达短期事件的脉冲余波特性，
        # 将脉冲发生瞬间向后自然延展 3 个交易日 (触发日 + 后延 3 日)。
        signal = raw_signal.replace(0.0, np.nan).ffill(limit=3).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"