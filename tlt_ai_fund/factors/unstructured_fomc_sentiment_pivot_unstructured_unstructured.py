import numpy as np
import pandas as pd

class UnstructuredFomcSentimentPivotFactor:
    """FOMC情绪突变反转脉冲因子 (unstructured/unstructured)

    逻辑: 捕捉美联储从鹰派突然转向鸽派(或相反)的超预期政策跳跃瞬间。
          1. 边际变化铁律: 绝对禁止使用情绪得分绝对值，使用5日动量(差分)捕捉政策预期的瞬间阶跃。
          2. 二阶导数铁律: 必须满足"极值+衰竭"双重确认。极值要求跳跃动量超2.0个标准差；衰竭要求预期必须跨越零轴(前值<=0且当前>0)，代表旧趋势彻底衰竭。
          3. 零值休眠铁律: 额外叠加动量二阶导数衰竭(diff(1)<=0)条件，避免在预期刚落地的首日剧烈博弈中接飞刀。仅在突变发生后的极短确认窗口（约4天）输出信号，常态严格为0.0。
    数据: fomc_sentiment
    触发: 5日变化量252日Z-Score > 2.0 (极值) + 情绪跨越0轴反转 (衰竭) + 动量二阶导回落 (防接飞刀)
    输出: 狙击手级别脉冲, 在鹰转鸽确认期输出 +1.0 (看多美债), 鸽转鹰确认期输出 -1.0 (看空美债), 其余为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_sentiment_pivot_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 常态休眠，全零初始化
        signal = pd.Series(0.0, index=data.index)

        # 检查数据是否存在
        if 'fomc_sentiment' not in data.columns:
            return signal

        # 填补非会议日的空值，维持阶梯数据的特性
        fomc = data['fomc_sentiment'].ffill()

        # 铁律3: 边际变化优先。计算 5 日变化量（提取预期突变动量，过滤低频绝对水位）
        delta_5 = fomc.diff(5)

        # 计算滚动 Z-Score (使用252日/约1年窗口，微小常数1e-6防止由于长期0值导致的除零错误)
        roll_std = delta_5.rolling(window=252, min_periods=63).std()
        zscore = delta_5 / (roll_std + 1e-6)

        # 铁律2: 二阶导数与衰竭防接飞刀
        # 对于阶梯型数据的跳跃，其单日跳跃动量的一阶导数 (差分) 在事件发生当天下跌最大，随后即刻归零 (平稳)。
        # 要求 acceleration <= 0 (多头) / >= 0 (空头)，过滤掉市场最动荡的首日，在动能稳固次日出手。
        acceleration = delta_5.diff(1)

        # 鹰转鸽反转 (看多美债)
        long_trigger = (
            (zscore > 2.0) &              # 脉冲极值: 鸽派预期爆发 (大于两倍标准差的尾部跳跃)
            (fomc.shift(5) <= 0.0) &      # 衰竭反转: 5天前仍为鹰派/中立预期，旧趋势已被打破
            (fomc > 0.0) &                # 衰竭反转: 当前实质性转为鸽派
            (acceleration <= 0.0)         # 动量回落: 跳跃不再加速攀升，预期落地发酵期
        )

        # 鸽转鹰反转 (看空美债)
        short_trigger = (
            (zscore < -2.0) &             # 脉冲极值: 鹰派预期爆发
            (fomc.shift(5) >= 0.0) &      # 衰竭反转: 5天前仍为鸽派/中立预期
            (fomc < 0.0) &                # 衰竭反转: 当前实质性转为鹰派
            (acceleration >= 0.0)         # 动量回落: 负向跳跃停止恶化 (负动量的差分 >= 0 即为趋稳)
        )

        # 处理可能出现的 NaN 逻辑值
        long_trigger = long_trigger.fillna(False)
        short_trigger = short_trigger.fillna(False)

        # 信号赋值: 仅在触发区间赋予强脉冲 +1.0 / -1.0
        signal[long_trigger] = 1.0
        signal[short_trigger] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"