import numpy as np
import pandas as pd

class OptionsGoldVolExhaustionFactor:
    """Options Gold Volatility Exhaustion Factor (unstructured/options)

    逻辑: 黄金隐含波动率(GVZ)是FICC领域极为重要的期权避险情绪代理指标。当其在短期内出现极端的脉冲式飙升(期权市场狂买避险)后一旦开始衰竭, 标志着宏观尾部风险极值已充分Price-in, 此时美联储往往被迫释放流动性或转鸽, 美债(TLT)作为确定性避险资产迎来主升浪。反之, 当波动率极度压缩后初现反弹, 预示着自满情绪终结, 容易遭遇紧缩冲击。
    数据: gvzcls (CBOE黄金ETF隐含波动率指数)
    触发: 多头脉冲: 5日边际变化量的252日Z-Score > 2.5 (期权恐慌极值) 且 当日diff < 0 (恐慌开始衰竭/回落)
          空头脉冲: 5日边际变化量的252日Z-Score < -2.0 (极度自满极值) 且 当日diff > 0 (自满破裂/波动率初升)
    输出: +1.0 (看多美债), -1.0 (看空美债), 常态为 0.0
    """

    def __init__(self, window=252, change_period=5):
        self.name = 'options_gold_vol_exhaustion'
        self.window = window
        self.change_period = change_period

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 初始信号必须全为 0.0 (狙击手级常态休眠)
        signal = pd.Series(0.0, index=data.index)

        # 缺失数据保护
        if 'gvzcls' not in data.columns:
            return signal

        gvz = data['gvzcls'].ffill()

        # 铁律3: 绝对禁止使用绝对水位, 计算期权波动率的边际脉冲变化
        gvz_change = gvz.diff(self.change_period)

        # 计算边际变化的滚动 Z-Score 识别极端事件
        roll_mean = gvz_change.rolling(window=self.window, min_periods=self.window//2).mean()
        roll_std = gvz_change.rolling(window=self.window, min_periods=self.window//2).std()

        # 避免除以零引起的 inf
        roll_std = roll_std.replace(0.0, np.nan)
        z_score = (gvz_change - roll_mean) / roll_std

        # 铁律2: 二阶导数衰竭条件 (绝对禁止接飞刀, 必须确认动量反转)
        daily_diff = gvz.diff(1)

        # 多头触发条件: 
        # 1. 情绪极其恐慌 (Z-Score > 2.5) 
        # 2. 恐慌情绪开始高位衰竭 (daily_diff < 0)
        long_condition = (z_score > 2.5) & (daily_diff < 0)

        # 空头触发条件:
        # 1. 情绪极其自满 (Z-Score < -2.0)
        # 2. 自满情绪破裂, 波动率开始抬头 (daily_diff > 0)
        short_condition = (z_score < -2.0) & (daily_diff > 0)

        # 脉冲赋值
        signal.loc[long_condition] = 1.0
        signal.loc[short_condition] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, change_period={self.change_period})"