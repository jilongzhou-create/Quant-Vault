import numpy as np
import pandas as pd

class UnstructuredFomcCurvePivotNonlinearFactor:
    """Unstructured Fomc Curve Pivot Nonlinear (unstructured/nonlinear)

    逻辑: 捕捉政策预期的极端突变与美债收益率曲线形态的非线性共振。由于 FOMC 情绪得分为低频阶梯状非结构化数据，本因子严格遵守边际变化铁律，使用三日动量捕捉鸽派/鹰派突变瞬间。同时结合短端利率(DGS2)的二阶极值衰竭或急剧暴跌，且要求曲线发生牛陡/熊平(Bull Steepening/Bear Flattening)印证，形成高胜率共振脉冲。非连续因子，狙击手级触发。
    数据: fomc_sentiment, dgs2, t10y2y
    触发: 
      多头: (FOMC鸽派突变 Z>2.0 且 DGS2暴跌 Z<-2.0 且 曲线变陡) OR (DGS2极度高估 Z>2.5 且 开始跌破3日均线衰竭 且 曲线变陡 且 FOMC边际偏鸽)
      空头: (FOMC鹰派突变 Z<-2.0 且 DGS2暴涨 Z>2.0 且 曲线变平) OR (DGS2极度低估 Z<-2.5 且 开始突破3日均线反转 且 曲线变平 且 FOMC边际偏鹰)
    输出: +1.0 看多美债(脉冲持留3天)，-1.0 看空美债，常态为 0.0
    """

    def __init__(self):
        self.name = 'unstructured_fomc_curve_pivot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 极静默 Series，遵守铁律1：零值休眠
        signal = pd.Series(0.0, index=data.index)
        
        # 1. 检查必备数据列 (禁止跨域引用无关CoreAnchor数据)
        required_cols = ['fomc_sentiment', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        df = data[required_cols].ffill()
        
        # 2. 边际变化计算 (严格遵守铁律3: 必须使用 diff 捕捉边际突变)
        # FOMC情绪得分突变 (数值域 [-1, 1], 1.0=极度鸽派, -1.0=极度鹰派)
        fomc_diff = df['fomc_sentiment'].diff(3)
        fomc_mean = fomc_diff.rolling(60).mean()
        # 极小值替代0方差，防止除0错误
        fomc_std = fomc_diff.rolling(60).std().replace(0, np.nan).fillna(1e-6)
        fomc_z = (fomc_diff - fomc_mean) / fomc_std
        
        # DGS2 短端利率的动量突发与水位极值 (美联储政策利率的最敏感前瞻指引)
        dgs2 = df['dgs2']
        dgs2_diff = dgs2.diff(3)
        dgs2_diff_mean = dgs2_diff.rolling(60).mean()
        dgs2_diff_std = dgs2_diff.rolling(60).std().replace(0, np.nan).fillna(1e-6)
        dgs2_diff_z = (dgs2_diff - dgs2_diff_mean) / dgs2_diff_std
        
        dgs2_level_mean = dgs2.rolling(120).mean()
        dgs2_level_std = dgs2.rolling(120).std().replace(0, np.nan).fillna(1e-6)
        dgs2_level_z = (dgs2 - dgs2_level_mean) / dgs2_level_std
        
        # 收益率曲线动量 (10Y-2Y, t10y2y_diff > 0 意味着变陡，结合DGS2下行即为经典的 Bull Steepening)
        t10y2y_diff = df['t10y2y'].diff(3)
        
        # 3. 触发逻辑组 A: 预期突变与动量共振 (Pivot Shock)
        # 多头脉冲: 情绪鸽派突变爆发 + 2年期美债收益率急剧暴跌 + 收益率曲线牛陡
        long_shock = (fomc_z > 2.0) & (dgs2_diff_z < -2.0) & (t10y2y_diff > 0)
        # 空头脉冲: 情绪鹰派突变爆发 + 2年期美债收益率急剧暴涨 + 收益率曲线熊平/倒挂加剧
        short_shock = (fomc_z < -2.0) & (dgs2_diff_z > 2.0) & (t10y2y_diff < 0)
        
        # 4. 触发逻辑组 B: 极值衰竭反转 (严格遵守铁律2: 二阶导数，防止在主跌浪接飞刀)
        # 多头抄底: DGS2上行定价加息至极度恐慌水位(Z>2.5) + 开始跌破3日均线(加息恐慌衰竭) + 曲线开始变陡 + FOMC配合边际偏鸽
        long_exhaustion = (dgs2_level_z > 2.5) & (dgs2 < dgs2.rolling(3).mean()) & (t10y2y_diff > 0) & (fomc_diff > 0)
        # 空头逃顶: DGS2下行定价降息至极度过防水位(Z<-2.5) + 开始突破3日均线(降息预期破灭反转) + 曲线开始变平 + FOMC配合边际偏鹰
        short_exhaustion = (dgs2_level_z < -2.5) & (dgs2 > dgs2.rolling(3).mean()) & (t10y2y_diff < 0) & (fomc_diff < 0)
        
        # 5. 聚合非线性共振信号
        raw_long = long_shock | long_exhaustion
        raw_short = short_shock | short_exhaustion
        
        # 6. 脉冲延长控制 (遵守铁律1: 控制 Target Trigger Rate 5%-15%)
        # 极端事件仅发生于单日，通过 rolling(3).max() 使信号存活3天，确保有效的波段捕捉而不成为连续型因子
        long_pulse = raw_long.rolling(3).max().fillna(0) > 0
        short_pulse = raw_short.rolling(3).max().fillna(0) > 0
        
        # 7. 信号赋值 ([-1.0, 1.0], 多空冲突时默认偏向多头美债的防御性配置)
        signal[long_pulse] = 1.0
        signal[short_pulse & ~long_pulse] = -1.0 
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"