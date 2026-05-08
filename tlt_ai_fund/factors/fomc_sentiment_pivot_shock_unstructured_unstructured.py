import numpy as np
import pandas as pd

class FomcSentimentPivotShockFactor:
    """FOMC情绪预期突变脉冲因子 (unstructured/NLP)

    逻辑: 捕捉美联储FOMC声明文本情绪(NLP得分)的极端边际跳跃。由于FOMC情绪得分是每年变动约8次的低频阶梯数据，
          其绝对的鹰鸽水位常伴随主跌浪，不具备直接指导意义。只有预期的瞬间大幅跳跃(Shock)才会引发美债重定价。
          同时因子内嵌了事件后发酵的"衰竭"条件：不在突变发生当天立刻追单(可能面临短线流动性踩踏和诱多/空)，
          而是等其边际动量短期均值跟上、冲击被市场初步消化发酵时再输出脉冲信号，规避左侧接飞刀风险。
    数据: fomc_sentiment (基于LLM的FOMC文本鹰鸽情绪得分，1.0为极度鸽派，-1.0为极度鹰派)
    触发: 5日情绪变化量的 Z-Score 绝对值 > 2.5 (极值) + 变化量不再超越其3日均值 (冲击动能衰竭)
    输出: +1.0 (极度鸽派突变，看多美债TLT), -1.0 (极度鹰派突变，看空美债TLT)
    """

    def __init__(self, lookback_window: int = 252, diff_window: int = 5):
        self.name = 'fomc_sentiment_pivot_shock'
        self.lookback = lookback_window
        self.diff_win = diff_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始 signal 必须为常态 0.0，满足铁律1: 零值休眠
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # 前向填充以处理非会议日的缺失值
        fomc = data['fomc_sentiment'].ffill()
        
        # 铁律3: 仅关注边际变化，绝不使用低频阶梯数据的绝对水平值
        # 使用 5 日变化量捕捉声明日及次日的情绪跳跃
        fomc_diff = fomc.diff(self.diff_win)
        
        # 计算 252 日滚动 Z-Score 确定跳跃的极端性
        # 分母加 1e-6 防止因长达数月内数据无变化导致的 std=0 报错
        roll_mean = fomc_diff.rolling(window=self.lookback, min_periods=21).mean()
        roll_std = fomc_diff.rolling(window=self.lookback, min_periods=21).std()
        z_score = (fomc_diff - roll_mean) / (roll_std + 1e-6)
        
        # 铁律2: 二阶导数/衰竭条件 (反接飞刀)
        # 突变发生当天，fomc_diff(当前值) 会远大于其自身的 3日均值。
        # 等待事件落地 2-3 天后，fomc_diff 将不再扩大，开始等于/小于其 3日均值。
        # 这确保了因子是在事件情绪第一波乱杀平息、真正的主趋势确立时上车。
        diff_ma3 = fomc_diff.rolling(3).mean()
        
        # 鸽派突变动能衰竭: 正向跳跃幅度不再创短期新高
        dovish_exhaustion = fomc_diff <= diff_ma3  
        # 鹰派突变动能衰竭: 负向跳跃幅度不再创短期新低
        hawkish_exhaustion = fomc_diff >= diff_ma3 
        
        # 信号组合：狙击手级极端脉冲
        # 鸽派突变：得分正向剧增 (Z > 2.5) -> 降息/扩表预期飙升 -> 衰竭确认后看多美债 (+1.0)
        dovish_shock = (z_score > 2.5) & (fomc_diff > 0) & dovish_exhaustion
        
        # 鹰派突变：得分负向剧减 (Z < -2.5) -> 加息/收紧预期飙升 -> 衰竭确认后看空美债 (-1.0)
        hawkish_shock = (z_score < -2.5) & (fomc_diff < 0) & hawkish_exhaustion
        
        # 赋值非零脉冲信号
        signal[dovish_shock] = 1.0
        signal[hawkish_shock] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback={self.lookback}, diff_win={self.diff_win})"