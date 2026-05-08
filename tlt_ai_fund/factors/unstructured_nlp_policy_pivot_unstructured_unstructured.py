import numpy as np
import pandas as pd

class UnstructuredNlpPolicyPivotFactor:
    """Unstructured NLP Policy Pivot Shock Factor

    逻辑: 融合美联储央行文本(fomc_sentiment)与新闻文本(usepuindxd经济政策不确定性)的极端边际跳跃，捕捉宏观预期发生突变的瞬间。此为典型的事件驱动脉冲因子，常态下NLP指标平缓波动（信号=0），仅在政策恐慌或情绪极端反转且动能开始衰竭时扣动扳机。
    数据: fomc_sentiment (央行鸽鹰态度，1.0=极度鸽派), usepuindxd (经济政策不确定性指数)
    触发: NLP情绪变化动量的 252日 Z-Score > 2.0 (极值条件)，且边际动能低于3日均值(二阶导数:防飞刀的衰竭确认)。满足条件后生成持续3日的脉冲。
    输出: +1.0 看多美债 (FOMC极其转鸽 / 不确定恐慌见顶回落带来的避险买盘)；-1.0 看空美债 (FOMC极其转鹰 / 不确定极度自满见底带来的通胀/紧缩预期重燃)
    """

    def __init__(self):
        self.name = 'unstructured_nlp_policy_pivot_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 数据校验
        if 'fomc_sentiment' not in data.columns or 'usepuindxd' not in data.columns:
            return signal

        # 获取数据并前向填充防止NaN干扰
        fomc = data['fomc_sentiment'].ffill()
        epu = data['usepuindxd'].ffill()

        # 铁律3: 边际变化 (Marginal Change Only)
        # 禁止使用绝对值。FOMC为双周(10日)跳变观察期，EPU为月度(21日)动能观察期
        fomc_diff = fomc.diff(10).fillna(0.0)
        epu_diff = epu.diff(21).fillna(0.0)

        # 计算 252日 Z-Score，设定合理的回溯窗口
        fomc_std = fomc_diff.rolling(window=252, min_periods=21).std().replace(0, np.nan)
        fomc_z = ((fomc_diff - fomc_diff.rolling(window=252, min_periods=21).mean()) / fomc_std).fillna(0.0)

        epu_std = epu_diff.rolling(window=252, min_periods=21).std().replace(0, np.nan)
        epu_z = ((epu_diff - epu_diff.rolling(window=252, min_periods=21).mean()) / epu_std).fillna(0.0)

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 极值条件: Z-Score > 2.0 (约5%尾部概率) 或 < -2.0
        # 衰竭条件: 当前变化量弱于过去3日均值（意味着动能高潮已过，避免在情绪主升浪中接飞刀）
        fomc_roll3 = fomc_diff.rolling(3).mean()
        # FOMC 大幅转鸽且开始企稳
        fomc_dovish_shock = (fomc_z > 2.0) & (fomc_diff <= fomc_roll3)
        # FOMC 大幅转鹰且开始企稳
        fomc_hawkish_shock = (fomc_z < -2.0) & (fomc_diff >= fomc_roll3)

        epu_roll3 = epu_diff.rolling(3).mean()
        # EPU 恐慌极值并见顶回落 -> 流动性危机结束，避险资金加速买入美债
        epu_panic_shock = (epu_z > 2.0) & (epu_diff < epu_roll3)
        # EPU 极度自满并见底抬头 -> 风险偏好重燃/通胀起步可能抛售美债
        epu_complacent_shock = (epu_z < -2.0) & (epu_diff > epu_roll3)

        # 逻辑合并
        buy_trigger = fomc_dovish_shock | epu_panic_shock
        sell_trigger = fomc_hawkish_shock | epu_complacent_shock

        # 铁律1: 零值休眠 (Sniper Pulse)
        # 仅在事件触发当天及随后的极短几天(3天)内产生脉冲信号，确保目标 Trigger Rate 维持在 5%-15%
        pulse_buy = buy_trigger.rolling(window=3, min_periods=1).max().fillna(0) == 1
        pulse_sell = sell_trigger.rolling(window=3, min_periods=1).max().fillna(0) == 1

        signal[pulse_buy] = 1.0
        signal[pulse_sell] = -1.0

        # 多空冲突发生时保守归零，不带方向赌博
        signal[pulse_buy & pulse_sell] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"