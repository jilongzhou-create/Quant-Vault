import numpy as np
import pandas as pd

class NlpContextualVolReversalFactor:
    """NlpContextualVolReversalFactor (volatility/unstructured)

    逻辑: 监控跨资产波动率(VIX与黄金GVZ)的极端狂飙，并在其开始衰竭时，利用FOMC情绪的边际动量来决定美债的抄底/反转方向。极端恐慌消退且美联储边际转鸽时看多(降息预期发酵)，边际转鹰时看空(恐慌溢价消退后回归高息主跌浪)。严格遵守极值+衰竭原则，常态休眠，脉冲触发。
    数据: vixcls, gvzcls (跨资产波动率), fomc_sentiment (非结构化文本情绪)
    触发: VIX或GVZ的252日 Z-Score > 2.5 (条件1:极值) AND 两者均向下击穿3日均线 (条件2:衰竭) AND FOMC情绪63日动量显著偏离 (条件3:边际方向确认)
    输出: +1.0 (做多美债脉冲), -1.0 (做空美债脉冲), 0.0 (无极端事件或无衰竭信号时休眠)
    """

    def __init__(self):
        self.name = 'nlp_contextual_vol_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号严格为 0.0，满足铁律1：常态休眠
        signal = pd.Series(0.0, index=data.index)

        # 数据缺失保护
        required_cols = ['vixcls', 'gvzcls', 'fomc_sentiment']
        for col in required_cols:
            if col not in data.columns:
                return signal

        # 填充缺失值，防前瞻
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        fomc = data['fomc_sentiment'].ffill()

        # ---------------------------------------------------------------------
        # 1. 波动率极值监控 (Volatility Extreme - 铁律2:第一阶段)
        # 使用 252 个交易日 (1年) 计算 Z-Score，定义长期经济学视角下的极端狂飙
        # ---------------------------------------------------------------------
        vix_mean = vix.rolling(window=252, min_periods=63).mean()
        vix_std = vix.rolling(window=252, min_periods=63).std()
        vix_z = (vix - vix_mean) / vix_std

        gvz_mean = gvz.rolling(window=252, min_periods=63).mean()
        gvz_std = gvz.rolling(window=252, min_periods=63).std()
        gvz_z = (gvz - gvz_mean) / gvz_std

        # 任一核心跨资产波动率突破 2.5 倍标准差，即标记为极端恐慌状态
        vol_extreme = (vix_z > 2.5) | (gvz_z > 2.5)
        
        # 将极值状态向后展期 3 天，形成"极值余震窗口"
        # 确保我们能在峰值过后的极短几天内捕捉到衰竭
        extreme_window = vol_extreme.rolling(window=3, min_periods=1).max() > 0

        # ---------------------------------------------------------------------
        # 2. 二阶导数衰竭确认 (Exhaustion Confirmation - 铁律2:第二阶段)
        # 绝对禁止直接买入极值！必须等待波动率有效击穿短期(3日)均线，确认恐慌动能衰竭
        # ---------------------------------------------------------------------
        vix_cooling = vix < vix.rolling(window=3, min_periods=1).mean()
        gvz_cooling = gvz < gvz.rolling(window=3, min_periods=1).mean()
        # 跨资产交叉确认：美股波动率和黄金波动率必须同步冷却，防止单资产骗线
        vol_exhaustion = vix_cooling & gvz_cooling

        # 核心触发器：处于极值余震窗口内，且波动率开始同步衰竭
        base_trigger = extreme_window & vol_exhaustion

        # ---------------------------------------------------------------------
        # 3. NLP 非结构化数据边际变化判定 (Marginal Change - 铁律3)
        # 绝对禁止使用绝对值！计算当前 FOMC 情绪相较于过去 63 天(一个宏观季度)的动量差值
        # ---------------------------------------------------------------------
        fomc_mean_63 = fomc.rolling(window=63, min_periods=21).mean()
        fomc_momentum = fomc - fomc_mean_63

        # ---------------------------------------------------------------------
        # 4. 信号脉冲生成
        # ---------------------------------------------------------------------
        # 设定 0.05 的阈值，要求政策情绪有明确的边际倾斜，过滤微小噪音
        # 当期多头脉冲：恐慌衰竭 + 美联储边际转鸽 (经济下行担忧，降息预期兑现，利好美债)
        raw_long = base_trigger & (fomc_momentum > 0.05)
        
        # 当期空头脉冲：恐慌衰竭 + 美联储边际转鹰 (加息恐慌释放完毕，回归高息基本面，利空美债)
        raw_short = base_trigger & (fomc_momentum < -0.05)

        # 严格执行 Sniper Pulse：仅在衰竭信号出现的首日(及连续符合条件的1-2天)触发
        # 利用 ~shift(1) 提取状态翻转的瞬间，确保即使在衰竭窗口内也不会变成连续常态因子
        pulse_long = raw_long & (~raw_long.shift(1).fillna(False))
        pulse_short = raw_short & (~raw_short.shift(1).fillna(False))

        signal[pulse_long] = 1.0
        signal[pulse_short] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"