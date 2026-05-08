import numpy as np
import pandas as pd

class DailyNewsPanicExhaustionFactor:
    """新闻恐慌极值衰竭因子 (panic_mean_reversion/unstructured)

    逻辑: 基于日度非结构化新闻经济政策不确定性指数(US Daily EPU)。当该指数在近期(10天)内逼近一年新高，代表市场陷入极端政策恐慌，
          当且仅当不确定性见顶回落且在当天向下突破5日均线时，表明恐慌情绪出现物理上的“二阶衰竭”，触发强看多脉冲(+1.0，精准抄底)。
          而在常态行情中，若日内新闻恐慌指数突现年度前5%分位级别的异动飙升，预示着宏观预期急剧恶化，触发看空脉冲(-1.0)。
    数据: usepuindxd (基于非结构化新闻文本提取的每日经济政策不确定性)
    输出: 1.0 (极度恐慌出现边际衰竭，强烈抄底), -1.0 (常态下恐慌突爆，看空恶化趋势), 0.0 (信号休眠期)
    触发条件: 严格限制在极端高位跌破短期均线瞬间或黑天鹅暴增之日，目标 Trigger Rate 控制在 5% - 10%。
    """

    def __init__(self):
        self.name = 'daily_news_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 如果缺少所需数据，直接休眠
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        # 前向填充处理非交易日或缺失数据
        epu = data['usepuindxd'].ffill()
        
        # 初始化零信号序列
        signal = pd.Series(0.0, index=data.index)
        
        # ---------------------------------------------------------
        # 1. 极端恐慌后的"见顶衰竭"逻辑 (看多信号 +1.0)
        # ---------------------------------------------------------
        # 提取过去252个交易日（约一年）的高点
        epu_252_high = epu.rolling(window=252, min_periods=63).max()
        
        # 判断近10日内是否达到或逼近（>95%）年度极端高点，这代表当前正处于"极度恐慌震荡期"
        extreme_threshold = epu_252_high * 0.95
        recent_extreme = (epu >= extreme_threshold).rolling(window=10, min_periods=1).max() > 0
        
        # 二阶导数衰竭铁律：严格要求动量必须反转
        # 条件A: 今日数值比昨日回落
        epu_falling = epu.diff(1) < 0
        # 条件B: 短期内向下击穿过去5个交易日的移动均线 (标志着上行趋势打破)
        epu_5d_mean = epu.rolling(window=5, min_periods=2).mean()
        below_5d_mean = epu < epu_5d_mean
        
        # 脉冲铁律控制：仅在跌破5日均线的"瞬间"当天触发，防止处于衰竭期后连续输出+1.0
        breakdown_moment = epu.shift(1) >= epu_5d_mean.shift(1)
        
        # 最终多头脉冲条件
        long_cond = recent_extreme & epu_falling & below_5d_mean & breakdown_moment
        
        # ---------------------------------------------------------
        # 2. 常态期间的"恐慌突爆"逻辑 (看空信号 -1.0)
        # ---------------------------------------------------------
        # 不在极端高位，代表市场前期处于长牛或常态回归期
        normal_state = ~recent_extreme
        
        # 计算新闻不确定性的单日跳跃幅度
        epu_pct_change = epu.pct_change(1).fillna(0.0)
        
        # 动态黑天鹅阈值：过去一年单日增幅的95分位数
        high_surge_threshold = epu_pct_change.rolling(window=252, min_periods=63).quantile(0.95)
        
        # 恶化脉冲：突发性跃升超过动态分位数，且为了避免低基数效应的误触，硬性要求绝对涨幅 > 20%
        sudden_panic = (epu_pct_change > high_surge_threshold) & (epu_pct_change > 0.20)
        
        # 最终空头脉冲条件
        short_cond = normal_state & sudden_panic
        
        # ---------------------------------------------------------
        # 3. 信号灌装与合规输出
        # ---------------------------------------------------------
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"