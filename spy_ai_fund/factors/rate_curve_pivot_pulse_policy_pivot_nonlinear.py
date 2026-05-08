import numpy as np
import pandas as pd

class RateCurvePivotPulseFactor:
    """Rate Curve Pivot Pulse Factor (policy_pivot/nonlinear)

    逻辑: 捕捉美联储货币政策预期发生剧烈反转的极短窗口(Bull Steepening / Bear Flattening)。短端利率剧烈变动并主导曲线变陡或变平时，预示政策拐点。
    数据: dgs2 (2年期国债收益率), t10y2y (10年期与2年期利差)
    输出: +1.0 表示强烈看多(鸽派转向+抛压枯竭)，-1.0 表示看空(鹰派冲击+上攻乏力)，常态为0.0
    触发条件: dgs2的5日动量达到过去一年1.5倍标准差极值，伴随t10y2y同向剧烈变陡/变平，且当日动量发生反向(情绪衰竭)，预期 Trigger Rate 约 5%-10%。
    """

    def __init__(self, window: int = 252, lookback: int = 5, z_rate: float = 1.5, z_curve: float = 1.0):
        self.name = 'rate_curve_pivot_pulse'
        self.window = window        # 一年期交易日基准
        self.lookback = lookback    # 捕捉单周级别的剧烈预期重定价
        self.z_rate = z_rate        # 利率动量阈值，1.5代表年度尾部事件
        self.z_curve = z_curve      # 曲线变动阈值，验证短期主导结构

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 确保所需数据存在
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 计算 5 日边际动量 (捕捉阶段性的抢跑预期)
        dgs2_mom = dgs2.diff(self.lookback)
        t10y2y_mom = t10y2y.diff(self.lookback)
        
        # 计算动量的滚动 Z-Score 以识别偏离常态的突变
        dgs2_mom_mean = dgs2_mom.rolling(self.window).mean()
        dgs2_mom_std = dgs2_mom.rolling(self.window).std()
        dgs2_z = (dgs2_mom - dgs2_mom_mean) / (dgs2_mom_std + 1e-8)
        
        t10y2y_mom_mean = t10y2y_mom.rolling(self.window).mean()
        t10y2y_mom_std = t10y2y_mom.rolling(self.window).std()
        t10y2y_z = (t10y2y_mom - t10y2y_mom_mean) / (t10y2y_mom_std + 1e-8)
        
        # 计算单日变化以确认情绪/动能衰竭 (二阶导数铁律防接飞刀)
        dgs2_daily_diff = dgs2.diff(1)
        
        # 【看多脉冲】: 鸽派突变 (Bull Steepening + 预期动能衰竭)
        # 短端急剧下行(市场抢跑降息) -> 导致收益率曲线急速变陡 -> 当日下行放缓或止跌反弹(宣泄完毕)
        bull_pivot = (dgs2_z < -self.z_rate) & (t10y2y_z > self.z_curve) & (dgs2_daily_diff > 0)
        
        # 【看空脉冲】: 鹰派突变 (Bear Flattening + 预期动能衰竭)
        # 短端急剧上行(鹰派更长时间维持高息) -> 导致收益率曲线急速变平/倒挂 -> 当日上冲放缓(恐慌兑现, 市场流动性恶化开跌)
        bear_pivot = (dgs2_z > self.z_rate) & (t10y2y_z < -self.z_curve) & (dgs2_daily_diff < 0)
        
        # 填充脉冲信号
        signal.loc[bull_pivot] = 1.0
        signal.loc[bear_pivot] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, lookback={self.lookback}, z_rate={self.z_rate}, z_curve={self.z_curve})"