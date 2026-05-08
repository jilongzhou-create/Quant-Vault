import numpy as np
import pandas as pd

class UnstructuredFomcMacdExhaustionFactor:
    """FOMC Sentiment MACD Exhaustion Factor (unstructured/unstructured)

    逻辑: 针对 FOMC 鹰鸽情绪突变(低频阶梯信号)，利用 5/21日 MACD 将绝对水位转化为动能波。
          数学上，阶梯信号的 MACD(5,21) 极值必然出现在事件后约 4-5 天。
          策略耐心等待这 4-5 天的"会后情绪漂移(Post-FOMC Drift)"完全消化并达到极值(Z > 1.5)，
          在二阶导数反转(动能衰竭)的瞬间触发为期 5 天的反向脉冲，精准执行"极值+衰竭"的抄底/逃顶。
    数据: fomc_sentiment
    触发: MACD 126日 Z-Score绝对值 > 1.5 + MACD首日反转(二阶导满足)。
    输出: 脉冲型，鹰派冲击枯竭看多(+1.0)，鸽派冲击枯竭看空(-1.0)。目标Trigger Rate控制在5-15%。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_macd_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 严格处理缺失列，非触发日必须输出全 0.0 (零值休眠铁律)
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 情绪得分：1.0=极度鸽派(看多美债), -1.0=极度鹰派(看空美债)
        # 前向填充保证非会议日是阶梯持平的，仅在会议日发生突变
        fomc = data['fomc_sentiment'].ffill()
        
        # 1. 边际变化铁律 (动能提取): MACD = 快线(5日，约1周) - 慢线(21日，约1月)
        # 绝对禁止直接使用 FOMC 绝对值！通过计算快慢线差值，将其提取为动能振荡器
        ema_fast = fomc.ewm(span=5, adjust=False).mean()
        ema_slow = fomc.ewm(span=21, adjust=False).mean()
        macd = ema_fast - ema_slow
        
        # 2. 极端冲击判定 (126日滚动窗口，约半年，包含约4次FOMC会议)
        macd_mean = macd.rolling(window=126, min_periods=21).mean()
        macd_std = macd.rolling(window=126, min_periods=21).std()
        macd_z = (macd - macd_mean) / (macd_std + 1e-6)
        
        # 3. 二阶导数铁律 (反接飞刀: 必须等待动能极值 + 动能首次出现拐点)
        
        # 鹰派动能极值衰竭 -> 会后恐慌抛售结束，反手看多美债(+1.0)
        # 条件：前一日仍处于极端鹰派下杀(Z < -1.5)，且MACD于今日触底反弹(即一阶导数为正，二阶导数出现拐点)
        bull_trigger = (macd_z.shift(1) < -1.5) & \
                       (macd > macd.shift(1)) & \
                       (macd.shift(1) <= macd.shift(2))
        
        # 鸽派动能极值衰竭 -> 会后乐观狂欢结束，反手看空美债(-1.0)
        # 条件：前一日仍处于极端鸽派冲高(Z > 1.5)，且MACD于今日触顶回落
        bear_trigger = (macd_z.shift(1) > 1.5) & \
                       (macd < macd.shift(1)) & \
                       (macd.shift(1) >= macd.shift(2))
        
        # 4. 零值休眠铁律 (脉冲扩展机制: 狙击手信号仅在极值衰竭后活跃 5 天)
        # 每次会议突变的衰竭点只会触发一次，将其保持 5 个交易日，确保 Trigger Rate 落在 5-15% 黄金区间
        bull_active = bull_trigger.rolling(window=5, min_periods=1).max() > 0
        bear_active = bear_trigger.rolling(window=5, min_periods=1).max() > 0
        
        # 初始赋值为 0.0，严格遵守非触发状态休眠
        signal = pd.Series(0.0, index=data.index)
        signal[bull_active] = 1.0
        signal[bear_active] = -1.0
        
        # 防御性消除可能的多空信号冲突重叠
        overlap = bull_active & bear_active
        signal[overlap] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"