import numpy as np
import pandas as pd

class NewsPanicReversionPulseFactor:
    """News-based Economic Policy Uncertainty Panic Reversion
    
    逻辑: EPU(Economic Policy Uncertainty)追踪新闻中关于政策不确定性的讨论量。由于美股(SPY)具有长牛和均值回归属性, 
          "不确定性落地(见顶回落)"是强烈的抄底信号(Buy)；而"平静期突然爆发不确定性"则是杀估值的冲击信号(Sell)。
    数据: usepuindxd (Daily Economic Policy Uncertainty Index, 基于新闻NLP的每日经济政策不确定性指数)
    输出: +1.0 (恐慌衰竭, 看多), -1.0 (突发恐慌, 看空), 0.0 (常态休眠)
    触发条件: Z-Score极值+短均线拐点触发脉冲, 预期 Trigger Rate 8% ~ 12%
    """

    def __init__(self, long_window=252, short_window=5, z_high=1.5, z_spike=1.5):
        self.name = 'news_panic_reversion_pulse'
        # 经济学参数: 252天(1年)宏观基准，5天(1周)短期动量，1.5代表突发事件的统计学高分位阈值
        self.long_window = long_window
        self.short_window = short_window
        self.z_high = z_high
        self.z_spike = z_spike

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 数据校验与缺省处理
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        # Daily EPU 包含大量基于新闻的噪音，使用3日指数移动平均提取核心情绪趋势
        epu_raw = data['usepuindxd'].ffill()
        epu = epu_raw.ewm(span=3, adjust=False).mean()
        
        # 1. 计算长期宏观基准(1年期)下的 Z-Score
        epu_mean = epu.rolling(self.long_window).mean()
        epu_std = epu.rolling(self.long_window).std() + 1e-6
        epu_z = (epu - epu_mean) / epu_std
        
        # 2. 短期动量与一阶导数(周度边际变化)
        epu_ma5 = epu.rolling(self.short_window).mean()
        
        # 5日变化量以捕捉突发冲击
        epu_diff = epu.diff(self.short_window)
        epu_diff_std = epu_diff.rolling(self.long_window).std() + 1e-6
        epu_diff_z = epu_diff / epu_diff_std
        
        # 默认输出全为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # =================================================================
        # LONG (+1.0): 极端恐慌衰竭 (抄底)
        # 严格遵守"二阶导数铁律"：绝不在高点接飞刀，必须等动量反转！
        # =================================================================
        # 过去一周内曾处于极端恐慌状态
        high_panic = epu_z.rolling(self.short_window).max() > self.z_high
        
        # 恐慌开始衰竭：EPU 跌破周度均线，且今日数值下降
        exhaustion = (epu < epu_ma5) & (epu.diff() < 0)
        
        # 零值休眠铁律：仅在恐慌衰竭发生的第一天触发脉冲
        exhaustion_pulse = exhaustion & (~exhaustion.shift(1).fillna(False))
        
        buy_cond = high_panic & exhaustion_pulse
        
        # =================================================================
        # SHORT (-1.0): 突发性恐慌冲击 (看空)
        # =================================================================
        # 冲击发生前，市场处于相对平静状态 (Z-Score < 0.5)
        # (此过滤条件极其关键，可避免在熊市恐慌期反复开空，导致被暴涨均值回归反杀)
        calm_state = epu_z.shift(self.short_window) < 0.5
        
        # 边际变化铁律：不看绝对值，看本周突发变动是否巨大
        spike = epu_diff_z > self.z_spike
        
        # 仅在突发冲击第一天触发脉冲
        spike_pulse = spike & (~(epu_diff_z.shift(1) > self.z_spike).fillna(False))
        
        sell_cond = calm_state & spike_pulse
        
        # 信号赋值
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(long_window={self.long_window}, short_window={self.short_window}, z_high={self.z_high}, z_spike={self.z_spike})"