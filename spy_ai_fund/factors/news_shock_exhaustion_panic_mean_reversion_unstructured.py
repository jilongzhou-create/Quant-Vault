import numpy as np
import pandas as pd

class NewsShockExhaustionFactor:
    """新闻恐慌冲击衰竭因子 (panic_mean_reversion/unstructured)

    逻辑: 专门针对美股市场设计的非结构化新闻情绪因子。媒体对经济政策的不确定性恐慌(EPU)通常呈脉冲式爆发。利用极值+二阶导数衰竭法则，当极度恐慌达到高位后，首次出现断崖式新闻降温(单日跌幅显著且跌破短均线)，表明引发恐慌的利空被市场消化，触发极其精准的抄底买点；反之，长牛平静期内一旦突发不确定性飙升，即为趋势恶化的早期负面脉冲。
    数据: [usepuindxd] (Economic Policy Uncertainty Index for US, 日频新闻文本不确定性)
    输出: +1.0 (极度新闻恐慌衰退，强烈看多), -1.0 (平静期黑天鹅突袭，看空恶化), 0.0 (常态休眠)
    触发条件: 满足近期极大离群(Z>1.2)且今日环比降温超10%触发多头。满足极度平静(Z<-1.0)且单日急飙超15%触发空头。预期 Trigger Rate 控制在 8%-12%。
    """

    def __init__(self):
        self.name = 'news_shock_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 数据缺失保护
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 获取非结构化政策不确定性序列并处理缺失值
        epu = data['usepuindxd'].ffill()
        
        # 初始化休眠的脉冲信号 (全0序列)
        signal = pd.Series(0.0, index=data.index)

        # --------------------------------------------------------
        # 1. 计算长期物理背景刻度：基于自然年(252交易日)度量恐慌环境
        # --------------------------------------------------------
        rolling_mean = epu.rolling(window=252, min_periods=60).mean()
        rolling_std = epu.rolling(window=252, min_periods=60).std()
        
        # 避免零除，使用微小常数保护
        z_score = (epu - rolling_mean) / (rolling_std + 1e-5)

        # --------------------------------------------------------
        # 2. 计算短期边际动能 (寻找预期发生突变的拐点)
        # --------------------------------------------------------
        # 单日相对环比变化率
        pct_change = epu.diff() / (epu.shift(1) + 1e-5)
        # 短期 5日平滑均线，用作破位验证的辅助确认
        ma5 = epu.rolling(window=5, min_periods=1).mean()

        # --------------------------------------------------------
        # 3. 多头脉冲狙击逻辑 (捕捉抄底买入的衰竭时刻，严格防接飞刀)
        # --------------------------------------------------------
        # 极度恐慌条件：过去 5 个交易日内，至少有一次摸到了历史高分位(Z > +1.2)
        recent_panic = z_score.rolling(window=5, min_periods=1).max() > 1.2
        
        # 恐慌衰竭条件(二阶导数防接飞刀)：恐慌新闻今天骤降(跌幅超 10%)
        panic_exhausted = pct_change < -0.10
        
        # 破位确认条件：向下确立跌破 5日均线 (拒绝悬在半空的高位震荡)
        break_trend_down = epu < ma5
        
        buy_condition = recent_panic & panic_exhausted & break_trend_down

        # --------------------------------------------------------
        # 4. 空头脉冲狙击逻辑 (捕捉长牛平静期内突然爆发的趋势恶化)
        # --------------------------------------------------------
        # 极度麻痹条件：过去 10 个交易日内，市场极度平静(缺乏不确定性, Z < -1.0)
        recent_calm = z_score.rolling(window=10, min_periods=1).min() < -1.0
        
        # 恶化突变条件：新闻层面的不确定性突然炸量(飙升超 15%)
        sudden_shock = pct_change > 0.15
        
        # 破位确认条件：向上突破 5日短均线，确立新的一轮恐慌周期
        break_trend_up = epu > ma5

        sell_condition = recent_calm & sudden_shock & break_trend_up

        # --------------------------------------------------------
        # 5. 信号拼装与离散输出
        # --------------------------------------------------------
        signal[buy_condition] = 1.0
        signal[sell_condition] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"