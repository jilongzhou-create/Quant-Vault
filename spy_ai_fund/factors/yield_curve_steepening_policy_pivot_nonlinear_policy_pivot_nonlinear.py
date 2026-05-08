import numpy as np
import pandas as pd

class YieldCurveSteepeningPolicyPivotNonlinearFactor:
    """YieldCurveSteepeningPolicyPivotNonlinearFactor (policy_pivot/nonlinear)

    逻辑: 捕捉短端利率急剧下跌导致收益率曲线急速陡峭化的脉冲时刻(Bull Steepening)，这代表市场对美联储政策向鸽派发生剧变(抢跑降息)，产生流动性释放的看多脉冲；反之，短端急升导致曲线平坦化或倒挂加剧(Bear Flattening)，代表紧缩恐慌预期飙升，产生看空脉冲。
    数据: dgs2 (2年期美债收益率), t10y2y (10年期与2年期利差)
    输出: +1.0 看多 (鸽派突变/降息抢跑), -1.0 看空 (鹰派突变/紧缩恐慌), 0.0 常态休眠
    触发条件: 3日短端利率变动与利差变动均突破半年滚动1倍标准差极值，且当日动量继续确认该方向。预期Trigger Rate: 8% ~ 12%。
    """

    def __init__(self):
        self.name = 'yield_curve_steepening_policy_pivot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 数据字段校验
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            signal.name = self.name
            return signal

        # 填补节假日缺失值，保持物理意义的连续性
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # 1. 核心脉冲特征: 3日边际动量 (反映FOMC会议或宏观数据发布后极短期的预期重定价)
        dgs2_3d = dgs2.diff(3)
        t10y2y_3d = t10y2y.diff(3)

        # 2. 状态极值度量: 126日(约半年)滚动 Z-Score，识别是否属于近期宏观周期内的"突发预期剧变"
        roll_window = 126
        
        dgs2_3d_mean = dgs2_3d.rolling(window=roll_window, min_periods=30).mean()
        dgs2_3d_std = dgs2_3d.rolling(window=roll_window, min_periods=30).std()
        dgs2_z = (dgs2_3d - dgs2_3d_mean) / dgs2_3d_std.replace(0, np.nan)
        
        t10y2y_3d_mean = t10y2y_3d.rolling(window=roll_window, min_periods=30).mean()
        t10y2y_3d_std = t10y2y_3d.rolling(window=roll_window, min_periods=30).std()
        t10y2y_z = (t10y2y_3d - t10y2y_3d_mean) / t10y2y_3d_std.replace(0, np.nan)

        # 3. 日内动能确认特征 (过滤极值区间内的无效震荡日，要求爆发当天必须有实质性动能)
        dgs2_1d = dgs2.diff(1)
        t10y2y_1d = t10y2y.diff(1)

        # 4. 非线性交叉触发逻辑
        # 多头: Bull Steepening 脉冲 (鸽派/降息抢跑)
        # 逻辑: 短端收益率冲量暴跌(Z < -1.0) AND 曲线急速陡峭化(Z > 1.0) AND 今日继续跌超1个基点(确认无反弹)
        bull_pulse = (
            (dgs2_z < -1.0) & 
            (t10y2y_z > 1.0) & 
            (dgs2_1d < -0.01) & 
            (t10y2y_1d > 0.0)
        )

        # 空头: Bear Flattening 脉冲 (鹰派/加息恐慌)
        # 逻辑: 短端收益率冲量暴涨(Z > 1.0) AND 曲线急速倒挂/平坦化(Z < -1.0) AND 今日继续涨超1个基点(确认动能)
        bear_pulse = (
            (dgs2_z > 1.0) & 
            (t10y2y_z < -1.0) & 
            (dgs2_1d > 0.01) & 
            (t10y2y_1d < 0.0)
        )

        # 5. 写入纯脉冲信号
        signal[bull_pulse] = 1.0
        signal[bear_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"