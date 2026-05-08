import numpy as np
import pandas as pd

class PolicyPivotYieldCurveNonlinearFactor:
    """Policy Pivot Yield Curve Nonlinear Pulse Factor (policy_pivot/nonlinear)

    逻辑: 捕捉市场对美联储政策预期的剧变（流动性冲量）。通过短端利率（DGS2）的动量极值与期限利差（T10Y2Y）的动量极值进行非线性交叉。当DGS2急跌且曲线急剧变陡（Bull Steepening）时，意味着流动性宽松预期突变，市场抢跑降息，强烈看多美股；当DGS2急升且曲线急剧变平/倒挂（Bear Flattening）时，意味着紧缩预期升温或Higher for Longer，杀估值看空美股。
    数据: dgs2, t10y2y
    输出: +1.0 (鸽派转向/Bull Steepening), -1.0 (鹰派突变/Bear Flattening), 0.0 (常态休眠)
    触发条件: DGS2的5日变化量Z-Score < -1.8 且 T10Y2Y 5日变化量 Z-Score > 1.5 触发看多；反之触发看空。预期Trigger Rate 5%-10% (脉冲信号)。
    """

    def __init__(self):
        self.name = 'policy_pivot_yield_curve_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 默认输出全0休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        # 数据缺失检查
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            signal.name = self.name
            return signal
            
        # 提取数据并处理缺失值
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 边际变化铁律: 计算短期(5天/一周)动量变化量
        dgs2_diff = dgs2.diff(5)
        t10y2y_diff = t10y2y.diff(5)
        
        # 动态波动率标准化 (Z-Score计算, 窗口为1个交易年252天)
        # 使用 min_periods=60 保证初期有一定的有效数据
        dgs2_mean = dgs2_diff.rolling(window=252, min_periods=60).mean()
        dgs2_std = dgs2_diff.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        dgs2_zscore = (dgs2_diff - dgs2_mean) / dgs2_std
        
        t10y2y_mean = t10y2y_diff.rolling(window=252, min_periods=60).mean()
        t10y2y_std = t10y2y_diff.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        t10y2y_zscore = (t10y2y_diff - t10y2y_mean) / t10y2y_std
        
        # 非线性特征交叉触发逻辑
        # 强看多触发点: Bull Steepening (短端极剧下行 + 曲线剧烈变陡) -> 美联储意外"放水"/降息预期
        bull_steepening = (dgs2_zscore < -1.8) & (t10y2y_zscore > 1.5)
        
        # 强看空触发点: Bear Flattening (短端极剧飙升 + 曲线剧烈平坦化) -> 通胀超预期/加息预期强化
        bear_flattening = (dgs2_zscore > 1.8) & (t10y2y_zscore < -1.5)
        
        # 填充脉冲信号
        signal[bull_steepening] = 1.0
        signal[bear_flattening] = -1.0
        
        # 处理可能的缺失值并命名
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"