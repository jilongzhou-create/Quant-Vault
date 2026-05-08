import numpy as np
import pandas as pd

class PolicyPivotCurveShockFactor:
    """政策预期突变脉冲因子 (Policy Pivot Shock)

    逻辑: 捕捉美联储政策预期的极端跳跃瞬间。短端利率暴跌(降息预期骤升)且收益率曲线急剧变陡(Bull Steepening)时，如果跌势开始衰竭，则触发看多美债的脉冲。相反，超预期加息导致短端飙升与曲线熊平(Bear Flattening)且涨势衰竭时，看空美债。这是纯正的边际变化+二阶导数狙击手因子。
    数据: dgs2 (2年期美债收益率，最敏锐的政策预期前瞻指标), t10y2y (10年-2年利差，曲线形态)
    触发: dgs2的5日边际变化Z-Score < -2.5 且 t10y2y变化Z-Score > 2.0，加上短端日内跌幅小于3日均值(衰竭) -> +1.0
    输出: [-1.0, 1.0] 的极值脉冲信号，常态休眠为 0.0
    """

    def __init__(self, z_score_window=252, momentum_window=5, exhaustion_window=3):
        self.name = 'policy_pivot_curve_shock'
        self.z_score_window = z_score_window
        self.momentum_window = momentum_window
        self.exhaustion_window = exhaustion_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 核心铁律1: 零值休眠，非触发日信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)

        # 检查数据完备性
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return signal

        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # 核心铁律3: 边际变化 (Marginal Change Only)
        # 绝对禁止直接看收益率水位或倒挂与否，必须看预期的边际改变突波
        dgs2_diff = dgs2.diff(self.momentum_window)
        t10y2y_diff = t10y2y.diff(self.momentum_window)

        # 计算滚动 Z-Score 捕捉极端事件
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(self.z_score_window).mean()) / dgs2_diff.rolling(self.z_score_window).std()
        t10y2y_z = (t10y2y_diff - t10y2y_diff.rolling(self.z_score_window).mean()) / t10y2y_diff.rolling(self.z_score_window).std()

        # 核心铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 用短周期单日变化的均值收敛来判断趋势是否已过极度疯狂期
        dgs2_daily_diff = dgs2.diff(1)
        dgs2_daily_diff_ma = dgs2_daily_diff.rolling(self.exhaustion_window).mean()
        
        # 多头买点衰竭条件: 跌势减缓 (今日单日变化大于过去3日平均，由于是负数，即代表暴跌收敛)
        bull_exhaustion = dgs2_daily_diff > dgs2_daily_diff_ma
        
        # 空头卖点衰竭条件: 涨势减缓 (今日单日变化小于过去3日平均)
        bear_exhaustion = dgs2_daily_diff < dgs2_daily_diff_ma

        # 多头脉冲条件:
        # 1. 极度鸽派突变 (短端暴跌, Z < -2.5)
        # 2. Bull Steepening 确认 (利差急速走阔变陡, Z > 2.0)
        # 3. 恐慌衰竭 (暴跌动能收敛)
        buy_cond = (dgs2_z < -2.5) & (t10y2y_z > 2.0) & bull_exhaustion

        # 空头脉冲条件:
        # 1. 极度鹰派惊吓 (短端暴涨, Z > 2.5)
        # 2. Bear Flattening 确认 (利差急速收缩甚至深度倒挂, Z < -2.0)
        # 3. 鹰派恐慌衰竭
        sell_cond = (dgs2_z > 2.5) & (t10y2y_z < -2.0) & bear_exhaustion

        buy_cond = buy_cond.fillna(False)
        sell_cond = sell_cond.fillna(False)

        signal.loc[buy_cond] = 1.0
        signal.loc[sell_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_score_window={self.z_score_window}, momentum_window={self.momentum_window}, exhaustion_window={self.exhaustion_window})"