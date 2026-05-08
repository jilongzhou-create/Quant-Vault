import numpy as np
import pandas as pd

class NewsEpuPanicExhaustionFactor:
    """新闻经济政策不确定性恐慌衰竭脉冲 (panic_mean_reversion/unstructured)

    逻辑: 每日新闻经济政策不确定性(EPU)指数反映了宏观环境的非结构化情绪。当不确定性处于历史极端高位后见顶回落(跌破短期均线)，往往对应风险偏好修复，产生抄底买点(+1.0)；而当EPU从低位突然爆发但未达极值时，代表不确定性加剧、趋势恶化，产生看空信号(-1.0)。
    数据: usepuindxd (美国每日新闻经济政策不确定性指数)
    输出: +1.0 看多(极度恐慌衰竭), -1.0 看空(不确定性突发), 常态为 0.0
    触发条件: 
      - 看多(+1.0): 过去5天EPU 3日均值Z-Score > 1.25(极端恐慌)，且当日EPU跌破5日均线(恐慌动量逆转的瞬间)
      - 看空(-1.0): 前一日Z-Score < 0.5，当日Z-Score突破0.5但<1.25(轻微恐慌)，且EPU单日飙升>10%(预期瞬变脉冲)
      预期 Trigger Rate 控制在 5-15% 内。
    """

    def __init__(self, z_window=252, extreme_z=1.25, short_z_high=1.25, short_z_low=0.5, surge_pct=0.10):
        self.name = 'news_epu_panic_exhaustion'
        self.z_window = z_window
        self.extreme_z = extreme_z
        self.short_z_high = short_z_high
        self.short_z_low = short_z_low
        self.surge_pct = surge_pct

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 如果缺少所需数据，直接返回 0.0
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        epu = data['usepuindxd'].ffill()
        
        # 消除极端日频噪音，计算3日均值用于Z-score计算基准
        epu_ma3 = epu.rolling(window=3).mean()
        
        # 计算 252 日 Z-Score 衡量相对不确定性水位
        roll_mean = epu_ma3.rolling(window=self.z_window).mean()
        roll_std = epu_ma3.rolling(window=self.z_window).std()
        
        # 避免除以 0
        roll_std = roll_std.replace(0, np.nan)
        z_score = (epu_ma3 - roll_mean) / roll_std
        
        # --- 看多信号 (+1.0) : 极端恐慌 + 见顶回落脉冲 (二阶导数铁律) ---
        # 极值条件: 过去 5 个交易日内，Z-Score 曾经突破过极端阈值
        extreme_panic = z_score.rolling(window=5).max().shift(1) > self.extreme_z
        
        # 衰竭脉冲: 跌破 5日均线 (确认预期见顶回落，仅在交叉瞬间触发)
        epu_ma5 = epu.rolling(window=5).mean()
        pulse_down = (epu.shift(1) >= epu_ma5.shift(1)) & (epu < epu_ma5)
        
        long_signal = extreme_panic & pulse_down
        
        # --- 看空信号 (-1.0) : 轻度恐慌爆发脉冲 (边际变化铁律) ---
        # 爆发跳跃条件: 前一日平静(Z<0.5)，今日突然突破(Z>=0.5)，但未达极端(Z<1.25)
        z_breakout = (z_score.shift(1) < self.short_z_low) & (z_score >= self.short_z_low) & (z_score < self.short_z_high)
        
        # 且单日飙升明显, 预期恶化
        epu_surge = (epu / epu.shift(1).replace(0, np.nan) - 1) > self.surge_pct
        
        short_signal = z_breakout & epu_surge
        
        # --- 信号合成 ---
        signal = pd.Series(0.0, index=data.index)
        signal[long_signal] = 1.0
        signal[short_signal] = -1.0
        
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_window={self.z_window}, extreme_z={self.extreme_z})"