import numpy as np
import pandas as pd

class RealYieldPivotPulseFactor:
    """Real Yield Pivot Pulse Factor (policy_pivot/nonlinear)

    逻辑: 实际利率(TIPS)在5天内剧烈下行且通胀预期未崩塌时, 代表市场定价美联储进行非衰退型的"预防性鸽派转向", 驱动估值扩张。
    数据: dfii5(5年期实际收益率), t5yie(5年期通胀预期)
    输出: +1.0 看多(流动性脉冲), -1.0 看空(紧缩冲击), 平时 0.0
    触发条件: 实际利率处于中期中高位且5日暴降超12bps(通胀预期跌幅<10bps); 预期Trigger Rate约8%
    """

    def __init__(self):
        self.name = 'real_yield_pivot_pulse_policy_pivot_nonlinear'
        self.z_window = 126         # 半年交易日, 确定实际利率的中期相对水位
        self.chg_window = 5         # 一周交易日, 捕捉极短期的边际冲量
        self.chg_threshold = 0.12   # 12个基点(0.12%), 利率市场的剧烈波动阈值
        self.inf_threshold = -0.10  # 容忍的最大通胀预期降幅, 排雷"通缩型衰退倒逼降息"

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 检查必要数据列
        if 'dfii5' not in data.columns or 't5yie' not in data.columns:
            return signal

        dfii5 = data['dfii5'].ffill()
        t5yie = data['t5yie'].ffill()

        # 计算实际利率的中期 Z-Score 水位 (评估政策原本是紧还是松)
        mean_dfii5 = dfii5.rolling(window=self.z_window, min_periods=21).mean()
        std_dfii5 = dfii5.rolling(window=self.z_window, min_periods=21).std()
        std_dfii5 = std_dfii5.replace(0, np.nan)  # 避免除以 0
        dfii5_z = (dfii5 - mean_dfii5) / std_dfii5

        # 计算一周内的边际动量变化 (捕获预期的突变)
        dfii5_chg = dfii5.diff(self.chg_window)
        t5yie_chg = t5yie.diff(self.chg_window)

        # 非线性交叉 1: 多头买点 (金发姑娘式政策转向)
        # 条件: 此前实际利率偏高(存在转鸽空间) + 实际利率剧烈暴跌 + 通胀预期坚挺(非衰退逼爆)
        long_cond = (dfii5_z > 0.0) & (dfii5_chg < -self.chg_threshold) & (t5yie_chg > self.inf_threshold)

        # 非线性交叉 2: 空头卖点 (鹰派流动性紧缩冲击)
        # 条件: 此前实际利率偏低(舒适区) + 实际利率剧烈暴涨(市场遭流动性抽水)
        short_cond = (dfii5_z < 0.0) & (dfii5_chg > self.chg_threshold)

        # 狙击手脉冲过滤: 只在状态发生反转的瞬间(第一天)触发，禁止连续输出
        long_pulse = long_cond & (~long_cond.shift(1).fillna(False))
        short_pulse = short_cond & (~short_cond.shift(1).fillna(False))

        signal[long_pulse] = 1.0
        signal[short_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"