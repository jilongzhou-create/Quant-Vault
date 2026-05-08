import numpy as np
import pandas as pd

class DovishSteepeningPulseFactor:
    """Dovish Steepening Pulse (policy_pivot/nonlinear)

    逻辑: 捕捉美联储预期边际突变引发的债券市场重定价冲量。当2年期美债收益率短线急跌(市场抢跑降息)、收益率曲线牛市变陡且高收益债信用利差收窄时, 反映“金发姑娘”式的流动性宽松预期(非硬着陆恐慌), 产生看多脉冲；反之, 短端利率急升引发熊平且信用环境恶化时, 产生看空脉冲。
    数据: dgs2 (2年期国债收益率), t10y2y (期限利差), bamlh0a0hym2 (高收益债利差)
    输出: 满足鸽派变陡条件时输出 +1.0, 满足鹰派紧缩条件时输出 -1.0, 默认 0.0
    触发条件: 5日内 DGS2 降幅超8个基点 且 T10Y2Y 变陡超2个基点 且 信用利差收窄。只在条件达成的瞬间触发脉冲, 并维持3天以保证 5%-15% 的目标 Trigger Rate。
    """

    def __init__(self):
        self.name = 'dovish_steepening_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 数据校验与预处理
        required_cols = ['dgs2', 't10y2y', 'bamlh0a0hym2']
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        df = data[required_cols].ffill()

        # 2. 计算边际动量变化 (5日窗口捕获政策预期的短期脉冲)
        dgs2_5d = df['dgs2'].diff(5)
        t10y2y_5d = df['t10y2y'].diff(5)
        hy_5d = df['bamlh0a0hym2'].diff(5)

        # 3. 核心物理法则：条件交叉
        # 看多脉冲：预期降息(短端急降) + 曲线变陡 + 软着陆(信用利差收窄, 排除硬着陆衰退引发的接飞刀)
        bull_event = (dgs2_5d < -0.08) & (t10y2y_5d > 0.02) & (hy_5d < 0.0)
        
        # 看空脉冲：预期加息或higher-for-longer(短端急升) + 曲线变平 + 信用环境恶化
        bear_event = (dgs2_5d > 0.08) & (t10y2y_5d < -0.02) & (hy_5d > 0.0)

        # 4. 脉冲提取 (仅在预期剧变的突发瞬间触发)
        bull_pulse = bull_event & ~bull_event.shift(1).fillna(False)
        bear_pulse = bear_event & ~bear_event.shift(1).fillna(False)

        # 5. 信号适度延展以防休眠率过高，确保 Trigger Rate 达标 (维持3天短线脉冲)
        bull_signal = bull_pulse.rolling(window=3, min_periods=1).max() > 0
        bear_signal = bear_pulse.rolling(window=3, min_periods=1).max() > 0

        # 6. 生成最终信号
        signal = pd.Series(0.0, index=data.index)
        signal[bull_signal] = 1.0
        signal[bear_signal] = -1.0

        # 冲突处理(极端异常数据兜底)
        conflict = bull_signal & bear_signal
        signal[conflict] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"