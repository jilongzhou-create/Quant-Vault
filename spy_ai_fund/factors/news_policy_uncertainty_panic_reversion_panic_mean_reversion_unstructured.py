import numpy as np
import pandas as pd

class UnstructuredPanicMeanReversionFactor:
    """非结构化恐慌均值回归因子 (panic_mean_reversion/unstructured)

    逻辑: 结合两类核心NLP非结构化数据(FOMC情绪与基于新闻的经济政策不确定性EPU)。
          美股长牛属性下，恐慌极值见顶回落是绝佳买点。
          当EPU处于极端高位并开始猛烈回落(恐慌衰竭), 或FOMC情绪发生显著鸽派反转时, 形成强烈看多脉冲。
          当EPU在平静期突发飙升(突发黑天鹅), 或FOMC突然转鹰时, 形成看空脉冲。
    数据: fomc_sentiment (FOMC声明文本情绪), usepuindxd (基于NLP的经济政策不确定性)
    输出: 强烈看多(+1.0), 看空(-1.0), 常态0.0
    触发条件: FOMC边际突变当天及后2天, 或EPU极值衰竭/突发飙升时触发。预期Trigger Rate控制在 8% - 12%
    """

    def __init__(self):
        self.name = 'unstructured_panic_mean_reversion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        buy_mask = pd.Series(False, index=data.index)
        sell_mask = pd.Series(False, index=data.index)

        # 1. 结构A: FOMC Statement Sentiment Logic (低频阶梯状NLP数据)
        if 'fomc_sentiment' in data.columns:
            fomc = data['fomc_sentiment'].ffill()
            fomc_diff = fomc.diff(1)
            
            # Buy: 情绪显著转鸽(>0.25，跨越1/8个情绪光谱) 或 从鹰派(<0)正式翻转为鸽派(>=0)
            fomc_buy_trigger = (fomc_diff > 0.25) | ((fomc.shift(1) < 0.0) & (fomc >= 0.0) & (fomc_diff > 0.0))
            
            # Sell: 情绪显著转鹰(<-0.25) 或 从鸽派(>=0)正式翻转为鹰派(<0)
            fomc_sell_trigger = (fomc_diff < -0.25) | ((fomc.shift(1) >= 0.0) & (fomc < 0.0) & (fomc_diff < 0.0))
            
            # 使用向前 rolling max 使得脉冲在事件发生当天及后2天(共3天)有效，确保击中 5%-15% 的目标触发率
            fomc_buy = fomc_buy_trigger.rolling(window=3, min_periods=1).max().fillna(0).astype(bool)
            fomc_sell = fomc_sell_trigger.rolling(window=3, min_periods=1).max().fillna(0).astype(bool)
            
            buy_mask = buy_mask | fomc_buy
            sell_mask = sell_mask | fomc_sell

        # 2. 结构B: Economic Policy Uncertainty Logic (高频连续NLP数据)
        if 'usepuindxd' in data.columns:
            epu = data['usepuindxd'].ffill()
            
            # EPU单日新闻噪音极大，使用3日平滑获取核心趋势
            epu_smooth = epu.rolling(window=3, min_periods=1).mean()
            
            # 计算126日(半年)滚动Z-Score，识别宏观状态极值
            epu_mean = epu_smooth.rolling(window=126, min_periods=21).mean()
            epu_std = epu_smooth.rolling(window=126, min_periods=21).std()
            epu_z = (epu_smooth - epu_mean) / (epu_std + 1e-6)
            
            # 极值恐慌衰竭 (Buy): 
            # 满足二阶导数铁律: 昨日Z-Score处于极端恐慌尾部(>1.5, 约前6.6%)，且近3天Z-Score猛烈回落(diff < -0.5)
            epu_buy = (epu_z.shift(1) > 1.5) & (epu_z.diff(3) < -0.5)
            
            # 突发恐慌飙升 (Sell):
            # 昨日处于平静期(Z < 0.5)，且近3天Z-Score突发猛烈飙升(diff > 1.5, 通常为突发黑天鹅/政策惊吓)
            epu_sell = (epu_z.shift(1) < 0.5) & (epu_z.diff(3) > 1.5)
            
            buy_mask = buy_mask | epu_buy
            sell_mask = sell_mask | epu_sell

        # 信号合成: 冲突时多头优先 (迎合美股长牛均值回归属性)
        signal[sell_mask] = -1.0
        signal[buy_mask] = 1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"