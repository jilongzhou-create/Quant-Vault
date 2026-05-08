import numpy as np
import pandas as pd

class GoldVolShockExhaustionFactor:
    """黄金期权波动率恐慌脉冲衰竭因子 (unstructured/options)

    逻辑: 黄金隐含波动率(gvzcls)是一周内避险情绪和实际利率预期的核心代理指标。当 gvzcls 
          在短时间内发生极端的向上突变（避险/流动性恐慌），且该恐慌开始展现微观动量衰竭
          （回落低于3日均值）时，通常意味着央行干预预期上升或最坏尾部风险Price-in，
          长端美债(TLT)将迎来极强的修复性买盘。相反，当恐慌暴降并反转，意味着极端Risk-On
          导致美债遭到猛烈抛售。
    数据: gvzcls (CBOE黄金ETF隐含波动率)
    触发: 5日变化量的一年滚动 Z-Score > 2.0 (或 < -2.0) AND 价格日内动量衰竭 (反穿3日均线)
    输出: +1.0 (恐慌见顶衰竭，看多美债) / -1.0 (避险情绪崩溃反穿，看空美债) / 0.0 (常态休眠)
    """

    def __init__(self, chg_window=5, z_window=252, z_threshold=2.0, exhaust_window=3):
        self.name = 'gold_vol_shock_exhaustion'
        self.chg_window = chg_window
        self.z_window = z_window
        self.z_threshold = z_threshold
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'gvzcls' not in data.columns:
            return signal
            
        # 清洗与前向填充数据，避免NaN导致的无效计算
        gvz = data['gvzcls'].ffill()
        
        # 核心铁律3: 边际变化 (捕捉预期突变的瞬间，而非绝对水位)
        gvz_chg = gvz.diff(self.chg_window)
        
        # 计算一年期滚动微观分布
        roll_mean = gvz_chg.rolling(window=self.z_window, min_periods=63).mean()
        roll_std = gvz_chg.rolling(window=self.z_window, min_periods=63).std()
        
        # 避免除以 0 的情况
        roll_std = roll_std.replace(0, np.nan)
        zscore = (gvz_chg - roll_mean) / roll_std
        
        # 核心铁律2: 二阶导数 (微观衰竭确认，绝不接飞刀)
        # 高位衰竭：VIX变体跌破近期短期均线，且单日动量翻负
        exhaustion_high = (gvz < gvz.rolling(self.exhaust_window).mean()) & (gvz.diff(1) < 0)
        
        # 低位反转：VIX变体升破近期短期均线，且单日动量翻正
        exhaustion_low = (gvz > gvz.rolling(self.exhaust_window).mean()) & (gvz.diff(1) > 0)
        
        # 核心铁律1: 零值休眠 (狙击手级脉冲触发)
        # 极度恐慌 + 开始衰竭 = 买入美债避险/修复
        bull_condition = (zscore > self.z_threshold) & exhaustion_high
        
        # 避险崩溃 + 开始反弹 = Risk-On开启抛售美债
        bear_condition = (zscore < -self.z_threshold) & exhaustion_low
        
        signal[bull_condition] = 1.0
        signal[bear_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(chg_window={self.chg_window}, z_threshold={self.z_threshold})"