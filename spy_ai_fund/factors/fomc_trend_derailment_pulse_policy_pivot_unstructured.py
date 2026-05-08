import numpy as np
import pandas as pd

class NewsPolicyUncertaintyPulseFactor:
    """NewsPolicyUncertaintyPulse (policy_pivot/unstructured)

    逻辑: 结合基于新闻的经济政策不确定性(EPU)与FOMC文本情绪。极端不确定性耗竭且美联储未处于极度鹰派时抄底(正脉冲)；在相对平静期突然爆发不确定性飙升且美联储未处于鸽派时看空(负脉冲)。
    数据: usepuindxd (日常EPU), fomc_sentiment (FOMC文本情绪)
    输出: +1.0 看多 (极度恐慌衰竭), -1.0 看空 (轻微恐慌突增), 0.0 观望
    触发条件: EPU Z-Score > 1.3 且动量转负 (买入); EPU Z-Score < 0.5 且 3日内飙升 > 1.0 std (卖出)。严格控制连发，只输出脉冲信号。
    """

    def __init__(self):
        self.name = 'news_policy_uncertainty_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 检查数据可用性
        req_cols = ['usepuindxd', 'fomc_sentiment']
        missing = [c for c in req_cols if c not in data.columns]
        if missing:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 2. 前向填充非结构化数据
        epu = data['usepuindxd'].ffill()
        fomc = data['fomc_sentiment'].ffill()

        # 3. 计算基于半年的动态基准水位
        window = 126
        epu_mean = epu.rolling(window=window, min_periods=21).mean()
        epu_std = epu.rolling(window=window, min_periods=21).std()
        
        # 防止除零错误
        epu_std = epu_std.replace(0, np.nan)
        epu_z = (epu - epu_mean) / epu_std

        # 4. 计算边缘变化与动量
        epu_mom_3d = epu.diff(3)
        epu_diff_1d = epu.diff(1)

        # 5. 脉冲触发条件判定
        
        # --- 抄底逻辑: 极端恐慌衰竭 (Second Derivative) ---
        # 恐慌曾处于极值水位
        high_epu = epu_z.rolling(window=3).max() > 1.3
        # 恐慌开始实打实地回落 (极值+衰竭)
        exhaustion = (epu_mom_3d < 0) & (epu_diff_1d < 0)
        # 流动性环境护航: 绝对禁止在联储极其鹰派时接飞刀
        fed_not_ultra_hawk = fomc > -0.5
        
        buy_raw = high_epu.shift(1).fillna(False) & exhaustion & fed_not_ultra_hawk
        # 脉冲休眠铁律: 禁止连续多日连续触发
        buy_cond = buy_raw & (~buy_raw.shift(1).fillna(False))

        # --- 看空逻辑: 相对平静期的恐慌跳升 ---
        # 常态/低波环境
        calm_epu = epu_z.rolling(window=3).max() < 0.5
        # 不确定性新闻的剧烈跳跃 (超出日常噪音)
        spike = (epu_mom_3d > 1.0 * epu_std.shift(1)) & (epu_diff_1d > 0)
        # 禁止在联储释放大量鸽派预期时做空 (防被 Fed Put 轧空)
        fed_not_ultra_dove = fomc < 0.2
        
        sell_raw = calm_epu.shift(1).fillna(False) & spike & fed_not_ultra_dove
        # 脉冲休眠铁律: 禁止连续多日连续触发
        sell_cond = sell_raw & (~sell_raw.shift(1).fillna(False))

        # 6. 生成 [-1.0, 1.0] 脉冲信号
        signal = pd.Series(0.0, index=data.index, name=self.name)
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0

        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"