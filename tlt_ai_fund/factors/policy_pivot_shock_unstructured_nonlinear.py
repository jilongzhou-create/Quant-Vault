import numpy as np
import pandas as pd

class PolicyPivotShockFactor:
    """Policy Pivot Shock Factor (unstructured/nonlinear)

    逻辑: 捕捉美联储政策预期的极端跳跃和短端利率的非线性共振。当FOMC情绪突变且短端利率(2年期)暴跌导致收益率曲线急剧牛陡时，表明市场预期急剧转向降息。为严格遵守防接飞刀铁律，要求动量指标停止恶化（二阶导衰竭）瞬间才产生买入/卖出脉冲。
    数据: fomc_sentiment (FOMC鹰鸽情绪得分), dgs2 (2年期美债收益率), t10y2y (10年-2年利差)
    触发: FOMC情绪5日变化Z-Score极端(>2.0鸽派/<-2.0鹰派) + DGS2急降/急升 + 收益率曲线剧烈形变 + 动量二阶导衰竭
    输出: 极端鸽派转向且衰竭看多脉冲(+1.0), 极端鹰派转向且衰竭看空脉冲(-1.0), 否则严格休眠(0.0)
    """

    def __init__(self, mom_window=5, z_window=252, extreme_z=2.0, curve_z=1.5):
        self.name = 'policy_pivot_shock_unstructured_nonlinear'
        self.mom_window = mom_window
        self.z_window = z_window
        self.extreme_z = extreme_z
        self.curve_z = curve_z

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['fomc_sentiment', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal

        # 铁律3: 边际变化 (绝对禁止直接输出绝对值)
        # 计算动量 (预期突变与曲线形变)
        fomc_mom = data['fomc_sentiment'].diff(self.mom_window)
        dgs2_mom = data['dgs2'].diff(self.mom_window)
        t10y2y_mom = data['t10y2y'].diff(self.mom_window)

        # 计算 Z-Score 识别极端事件脉冲点
        fomc_z = (fomc_mom - fomc_mom.rolling(self.z_window).mean()) / fomc_mom.rolling(self.z_window).std()
        dgs2_z = (dgs2_mom - dgs2_mom.rolling(self.z_window).mean()) / dgs2_mom.rolling(self.z_window).std()
        t10y2y_z = (t10y2y_mom - t10y2y_mom.rolling(self.z_window).mean()) / t10y2y_mom.rolling(self.z_window).std()

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 必须等待动量冲顶回落，禁止在主跌浪或主升浪直接触发
        # 1. 情绪突变动量开始衰竭 (不再继续加速)
        fomc_exhaustion_bull = fomc_mom.diff(1) <= 0
        fomc_exhaustion_bear = fomc_mom.diff(1) >= 0

        # 2. 短端利率剧烈波动开始降速 (当前1日变化速度相较于前3日均值减弱)
        dgs2_daily_change = data['dgs2'].diff(1)
        dgs2_3d_mean = data['dgs2'].diff(3) / 3
        # 跌势衰竭：日跌幅小于3日平均跌幅，或开始反弹
        dgs2_exhaustion_bull = dgs2_daily_change >= dgs2_3d_mean 
        # 涨势衰竭：日涨幅小于3日平均涨幅，或开始回落
        dgs2_exhaustion_bear = dgs2_daily_change <= dgs2_3d_mean 

        # 铁律1: 零值休眠 (Sniper Pulse) + 方法C: 非线性特征交叉
        # 多头触发脉冲: 极鸽派突变 + 短端极速下行 + 曲线牛陡 + 多重衰竭确认
        bull_condition = (
            (fomc_z > self.extreme_z) &
            (dgs2_z < -self.extreme_z) &
            (t10y2y_z > self.curve_z) &
            (fomc_exhaustion_bull) &
            (dgs2_exhaustion_bull)
        )

        # 空头触发脉冲: 极鹰派突变 + 短端极速上行 + 曲线熊平 + 多重衰竭确认
        bear_condition = (
            (fomc_z < -self.extreme_z) &
            (dgs2_z > self.extreme_z) &
            (t10y2y_z < -self.curve_z) &
            (fomc_exhaustion_bear) &
            (dgs2_exhaustion_bear)
        )

        signal.loc[bull_condition] = 1.0
        signal.loc[bear_condition] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(mom_window={self.mom_window}, z_window={self.z_window}, extreme_z={self.extreme_z})"