import numpy as np
import pandas as pd

class UnstructuredFomcPivotPulseExhaustionFactor:
    """FOMC情绪突变与动能衰竭脉冲因子 (unstructured/unstructured)

    逻辑: 美联储政策声明(fomc_sentiment)具有低频阶梯状特征，直接使用绝对值会导致信号连续且严重滞后。本因子聚焦于宏观预期的“边际突变”(5日变化量突破极值)，并严格遵守不接飞刀原则：强制要求“日度变化量向零轴回落(低于3日均值)”作为突变脉冲衰竭的确认条件。这完美避开了FOMC事件当天的无序震荡，在市场开始沿新宏观路径Price-in的T+1至T+2日发出精准顺势的狙击脉冲。
    数据: fomc_sentiment (FOMC文本鹰鸽情绪得分，低频阶梯数据)
    触发: 边际突变(Z-Score > 2.5) + 脉冲衰竭(1日变化量脱离极值，向3日均线均值回归)
    输出: 极端鸽派突变且衰竭时输出 +1.0 (看多TLT)，极端鹰派突变且衰竭时输出 -1.0 (看空TLT)，常态严格为0
    """

    def __init__(self):
        self.name = 'unstructured_fomc_pivot_pulse_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 常态下必须返回 0.0 序列
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖数据是否存在
        if 'fomc_sentiment' not in data.columns:
            return signal

        # 获取数据并前向填充，防止数据确实导致计算断层
        fomc = data['fomc_sentiment'].ffill()

        # 铁律3: 边际变化 (绝对禁止使用水平绝对值，必须用 .diff() 捕捉变化)
        # d1: 用于捕捉跳跃当天的瞬间动能
        # d5: 用于平滑单日跳跃，捕捉完整的议息周期的边际变化水位
        d1 = fomc.diff(1)
        d5 = fomc.diff(5)

        # 计算5日变化量的 252日(约1年) 滚动 Z-Score
        # 添加 1e-8 防止长期无会议期间标准差为0导致的除零报错
        roll_mean_d5 = d5.rolling(window=252, min_periods=60).mean()
        roll_std_d5 = d5.rolling(window=252, min_periods=60).std() + 1e-8
        z_d5 = (d5 - roll_mean_d5) / roll_std_d5

        # 铁律2: 二阶导数 (脉冲衰竭条件)
        # 计算1日变化量的3日滚动均值。
        # 阶梯数据跳跃时 d1 瞬间极大，随后立刻归零。
        # 归零时(T+1日)，d1(为0) 将小于 d1_mean3(包含昨日极大值的正数)。这就是极其精准的衰竭/确认特征！
        d1_mean3 = d1.rolling(window=3, min_periods=1).mean()

        # ---------------- 触发逻辑组装 ----------------
        
        # 多头触发脉冲 (鸽派突变看多美债)
        # 条件1 (极值): z_d5 > 2.5 (近期发生了极为罕见的鸽派突变)
        # 条件2 (衰竭): d1 < d1_mean3 (跳跃动能已衰竭，市场情绪跨越拐点进入消化期)
        # 条件3 (方向): d5 > 0 (确保真实方向确实是向着鸽派移动)
        long_cond = (z_d5 > 2.5) & (d1 < d1_mean3) & (d5 > 0)

        # 空头触发脉冲 (鹰派突变看空美债)
        # 条件1 (极值): z_d5 < -2.5 (近期发生了极为罕见的鹰派突变)
        # 条件2 (衰竭): d1 > d1_mean3 (跳跃动能衰竭，d1归0大于此前的负向跳跃均值)
        # 条件3 (方向): d5 < 0 (确保真实方向确实是向着鹰派移动)
        short_cond = (z_d5 < -2.5) & (d1 > d1_mean3) & (d5 < 0)

        # 铁律1: 狙击手级输出，仅在触发日赋值
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"