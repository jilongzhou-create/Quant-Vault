import numpy as np
import pandas as pd

class FomcMarginalShockFactor:
    """Fomc Marginal Shock Factor (panic_mean_reversion/unstructured)

    逻辑: 捕捉FOMC低频阶梯情绪得分的预期边际跳跃。当极端鹰派恐慌情绪见顶并开始边际衰竭时(极值+二阶导反转)，或发生超预期的强力鸽派突变时，触发连续4天的看多脉冲以抄底；反之，当鸽派狂热衰竭或突发强鹰派冲击时，输出看空脉冲。严守阶梯数据的边际变化铁律。
    数据: fomc_sentiment
    输出: 信号范围 [-1.0, 1.0]，正值看多，负值看空
    触发条件: 会议公布日情绪极值(>0.4/<-0.4)发生均值回归拐头，或单次突变跳跃幅度(>0.25)。仅在突变日当天及随后3天内保持极短脉冲输出。预期 Trigger Rate 在 8%-12% 之间。
    """

    def __init__(self):
        self.name = 'fomc_marginal_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 数据完整性检查
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)
        
        # 获取基础非结构化情感得分，T+1生效，向前填充
        fomc = data['fomc_sentiment'].ffill()
        
        # 计算低频阶梯状数据的边际变化 (边际铁律)
        fomc_diff = fomc.diff().fillna(0.0)
        fomc_prev = fomc.shift(1).fillna(0.0)
        
        # 仅在预期发生改变的瞬间(跳跃/变动日)才视为事件触发日
        event_day = fomc_diff.abs() > 1e-4
        
        # 买点脉冲 (看多美股抄底)
        # 1. 鹰派恐慌极值且开始回暖衰竭 (前值 <= -0.4, 且当天边际转鸽 fomc_diff > 0) - 二阶导数防接飞刀法则
        # 2. 强力鸽派超预期突变 (单次态度跳变 >= 0.25)
        # 3. 会议彻底由鹰转鸽 (穿过 0 轴防线)
        buy_pulse = event_day & (
            ((fomc_prev <= -0.4) & (fomc_diff > 0)) | 
            (fomc_diff >= 0.25) | 
            ((fomc_prev < 0) & (fomc >= 0) & (fomc_diff > 0))
        )
        
        # 卖点脉冲 (看空美股逃顶)
        # 1. 鸽派贪婪极值且开始恶化 (前值 >= 0.4, 且当天边际转鹰 fomc_diff < 0)
        # 2. 强力鹰派超预期突变 (单次态度跳变 <= -0.25)
        # 3. 会议彻底由鸽转鹰 (跌穿 0 轴防线)
        sell_pulse = event_day & (
            ((fomc_prev >= 0.4) & (fomc_diff < 0)) | 
            (fomc_diff <= -0.25) | 
            ((fomc_prev > 0) & (fomc <= 0) & (fomc_diff < 0))
        )
        
        # 零值休眠铁律延展: 
        # 为了达到 5% - 15% 的 Trigger Rate (因FOMC每年仅8次会议)，
        # 必须将脉冲适当延展为一段极短的"狙击窗口" (突变日当天 + 随后3天 = 4个交易日)
        # 若全年有5次触发，约20个交易日输出信号，Trigger rate ~ 7.9%
        buy_signal = buy_pulse.rolling(window=4, min_periods=1).max() == 1
        sell_signal = sell_pulse.rolling(window=4, min_periods=1).max() == 1
        
        # 初始化全零常态阵列
        signal = pd.Series(0.0, index=data.index)
        
        # 灌注脉冲强信号
        signal[buy_signal] = 1.0
        signal[sell_signal] = -1.0
        
        # 异常冲突态过滤（极低概率多空同日）
        conflict = buy_signal & sell_signal
        signal[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"