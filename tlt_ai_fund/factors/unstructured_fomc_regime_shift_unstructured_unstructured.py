import numpy as np
import pandas as pd

class UnstructuredFomcRegimeShiftFactor:
    """Unstructured Fomc Regime Shift Factor (unstructured/unstructured)

    逻辑: 捕捉美联储FOMC情绪突变后的趋势延续(Policy Pivot Shock)。
          极端鹰/鸽派声明发布当天，市场往往剧烈震荡(接飞刀风险极大)。
          本因子严格遵守二阶导数衰竭铁律：等待 FOMC 情绪变动衰竭(变动率归零，达到新的稳态平台)后，
          在其后的5个交易日释放顺势脉冲信号，参与机构调仓带来的确定性美债趋势，避开当天的无序波动。
    数据: fomc_sentiment (FOMC声明情绪得分，范围[-1, 1]，1为极鸽)
    触发: fomc_sentiment 单日变化量的 252日 Z-Score > 2.5 (预期极端跳跃) 
          + 随后变化量趋近于0 (二阶导衰竭，新预期落地)
    输出: 稳态后的5日内脉冲，鸽派突变衰竭后看多(+1.0)，鹰派突变衰竭后看空(-1.0)
    """

    def __init__(self):
        self.name = 'unstructured_fomc_regime_shift'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 数据缺失保护
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 核心基座：前向填充非会议日的数据，维持低频阶梯状特征
        sentiment = data['fomc_sentiment'].ffill()
        signal = pd.Series(0.0, index=data.index)

        # 铁律3: 边际变化 (Marginal Change) - 绝对禁止使用绝对水平值
        daily_change = sentiment.diff()

        # 动态波动率测度 (涵盖过去一年的完整FOMC周期)
        roll_mean = daily_change.rolling(window=252, min_periods=21).mean()
        
        # 设定 0.05 为经济学意义上的底噪阈值
        # FOMC sentiment 域宽为 2.0 (从-1到1)，0.05 代表 2.5% 的极微小语义变动，视作无意义噪音
        # 此操作同时避免长周期无会议期间(或连续不变)导致的除以零错误
        roll_std = daily_change.rolling(window=252, min_periods=21).std().clip(lower=0.05)

        # 计算边际变化的 Z-Score
        z_score = (daily_change - roll_mean) / roll_std

        # 极端突变检测 (Extreme Shock)
        dove_shock = z_score > 2.5   # 鸽派突变 (大幅向1.0移动)
        hawk_shock = z_score < -2.5  # 鹰派突变 (大幅向-1.0移动)

        # 铁律2: 二阶导数衰竭 (Exhaustion) - 严禁在波动当天接飞刀
        # 阶梯数据的衰竭表现为：突变后的下一天，变化量重新归零 (消除浮点数精度误差，设阈值为0.01)
        # 这意味着市场已经拿到了确定的 FOMC 新声明文本，预期重定价的"瞬间"已结束
        is_exhausted = daily_change.abs() < 0.01

        # 铁律1: 零值休眠与狙击手脉冲 (Sniper Pulse)
        # 仅在发生极端 Shock 之后的 5 个交易日内释放脉冲，控制 Trigger Rate 在 5%~15% 的黄金区间
        # 使用 shift(1) 确保使用的是历史发生的 shock，无未来函数
        recent_dove = dove_shock.shift(1).rolling(window=5).max().fillna(0) == 1
        recent_hawk = hawk_shock.shift(1).rolling(window=5).max().fillna(0) == 1

        # 生成最终信号：必须同时满足 "已完成衰竭(无边际变动)" 和 "处于极端事件窗口期"
        # 美债是正向 Carry 资产，极鸽派(利好)给 +1.0，极鹰派(利空)给 -1.0
        signal[is_exhausted & recent_dove] = 1.0
        signal[is_exhausted & recent_hawk] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"