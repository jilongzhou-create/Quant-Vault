import numpy as np
import pandas as pd

class UnstructuredPolicyPivotPulseFactor:
    """Unstructured Policy Pivot Pulse (policy_pivot/unstructured)

    逻辑: 结合高频新闻政策不确定性(USEPUINDXD)与低频美联储声明情绪(fomc_sentiment)。捕捉政策预期的瞬间反转：当不确定性突升或联储意外转鹰时，产生看空脉冲；当极度不确定性快速衰竭落地，或联储意外转鸽时，产生看多脉冲。
    数据: [usepuindxd, fomc_sentiment]
    输出: [-1.0, 1.0] 脉冲信号
    触发条件: PU Z-score 跳升或FOMC突发转鹰 -> -1.0; PU 从极值快速衰竭或FOMC突发转鸽 -> 1.0。信号维持3天，目标 Trigger Rate 5-15%。
    """

    def __init__(self):
        self.name = 'unstructured_policy_pivot_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        has_pu = 'usepuindxd' in data.columns
        has_fomc = 'fomc_sentiment' in data.columns
        
        if not has_pu and not has_fomc:
            signal.name = self.name
            return signal

        buy_mask = pd.Series(False, index=data.index)
        sell_mask = pd.Series(False, index=data.index)

        # 1. 结构外数据：美联储低频NLP情绪 (捕捉鸽鹰边际突变)
        if has_fomc:
            fomc = data['fomc_sentiment'].ffill()
            fomc_diff = fomc.diff()
            
            # 边际变化铁律: 联储边际转鸽 (救市/宽松预期突变) -> 强看多脉冲
            buy_mask = buy_mask | (fomc_diff > 0.25)
            
            # 边际变化铁律: 联储边际转鹰 (收紧预期突变) -> 趋势恶化看空
            sell_mask = sell_mask | (fomc_diff < -0.25)

        # 2. 结构外数据：基于新闻的高频经济政策不确定性 (极值耗尽与突发恐慌)
        if has_pu:
            pu = data['usepuindxd'].ffill()
            
            # 使用 60 天滚动窗口计算局部 Z-Score，反映近期所处的不确定性水位
            pu_mean = pu.rolling(window=60, min_periods=20).mean()
            pu_std = pu.rolling(window=60, min_periods=20).std()
            pu_z = (pu - pu_mean) / pu_std.replace(0, 1e-5)
            
            # 5天动量，捕捉趋势快速改变
            pu_z_diff = pu_z.diff(5)
            
            # 追踪过去 10 天的不确定性极值 (恐慌蓄水池)
            pu_z_high = pu_z.rolling(10).max()
            
            # 买入逻辑 (极值 + 衰竭): 政策不确定性此前极高(Z>2.0)，但近日迅速回落(下降>1.0)并跌破1.0，说明不确定性出尽，政策落地
            pu_buy = (pu_z_high > 2.0) & (pu_z_diff < -1.0) & (pu_z < 1.0)
            
            # 卖出逻辑 (轻微恐慌): 从常态(Z<1.0)突然跃升(上升>1.5)至高位(Z>2.0)，说明出现突发政策利空
            pu_sell = (pu_z > 2.0) & (pu_z_diff > 1.5) & (pu_z.shift(5) < 1.0)
            
            buy_mask = buy_mask | pu_buy
            sell_mask = sell_mask | pu_sell

        # 赋值信号
        signal[buy_mask] = 1.0
        signal[sell_mask] = -1.0
        
        # 冲突处理 (同日多空抵消)
        conflict = buy_mask & sell_mask
        signal[conflict] = 0.0
        
        # 零值休眠铁律: 将脉冲信号保持 3 天 (limit=2)，以达到 5%-15% 的 Trigger Rate 目标并覆盖极短的定价窗口
        signal = signal.replace(0.0, np.nan).ffill(limit=2).fillna(0.0)
        
        signal.name = self.name
        return signal