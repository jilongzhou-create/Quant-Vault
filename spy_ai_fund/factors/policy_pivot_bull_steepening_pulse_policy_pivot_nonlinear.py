import numpy as np
import pandas as pd

class PolicyPivotBullSteepeningPulseFactor:
    """政策转向与牛陡脉冲因子 (policy_pivot/nonlinear)

    逻辑: 捕捉美联储政策预期发生非线性剧变的极短窗口。当短端利率(2年期)剧烈下行代表市场抢跑宽松，叠加收益率曲线剧烈变陡(Bull Steepening)，是长牛美股的强力流动性催化剂；反之短端飙升与曲线扁平化则是紧缩恐慌，预示趋势恶化。
    数据: dgs2 (2年期国债收益率), t10y2y (10年减2年国债利差)
    输出: +1.0(抢跑宽松牛陡买入), -1.0(紧缩恐慌熊平卖出), 0.0(常态休眠)
    触发条件: 2年期美债5日跌幅破近半年 -2.0 Z-Score 且 期限利差5日走阔破 1.5 Z-Score，今日仍未反弹时触发看多。预期 Trigger Rate 5%-12%。
    """

    def __init__(self):
        self.name = 'policy_pivot_bull_steepening_pulse'
        self.window = 126  # 半年交易日窗口，用于计算动态均值和方差
        self.momentum_days = 5  # 5日动量捕捉短期的"剧烈变化"

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必要数据列
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            signal.name = self.name
            return signal
            
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 计算动量变化 (捕捉预期突变的冲量)
        dgs2_chg = dgs2.diff(self.momentum_days)
        t10y2y_chg = t10y2y.diff(self.momentum_days)
        
        # 计算滚动的 Z-Score (动态自适应高息与低息时代的波动率)
        dgs2_std = dgs2_chg.rolling(self.window, min_periods=20).std() + 1e-6
        dgs2_mean = dgs2_chg.rolling(self.window, min_periods=20).mean()
        dgs2_z = (dgs2_chg - dgs2_mean) / dgs2_std
        
        t10y2y_std = t10y2y_chg.rolling(self.window, min_periods=20).std() + 1e-6
        t10y2y_mean = t10y2y_chg.rolling(self.window, min_periods=20).mean()
        t10y2y_z = (t10y2y_chg - t10y2y_mean) / t10y2y_std
        
        # 每日边际变化，二阶导数防飞刀: 确认今日的冲量并未发生日内反转
        dgs2_diff1 = dgs2.diff(1)
        t10y2y_diff1 = t10y2y.diff(1)
        
        # 抄底看多脉冲：短端剧烈下行 (极度降息预期) AND 曲线剧烈陡峭化 (牛陡) AND 今日顺势未反弹
        long_cond = (
            (dgs2_z < -2.0) & 
            (t10y2y_z > 1.5) & 
            (dgs2_diff1 < 0) & 
            (t10y2y_diff1 > 0)
        )
        
        # 恶化看空脉冲：短端剧烈冲高 (极度加息恐慌) AND 曲线剧烈扁平化/倒挂 (熊平) AND 今日顺势未回落
        short_cond = (
            (dgs2_z > 2.0) & 
            (t10y2y_z < -1.5) & 
            (dgs2_diff1 > 0) & 
            (t10y2y_diff1 < 0)
        )
        
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, momentum_days={self.momentum_days})"