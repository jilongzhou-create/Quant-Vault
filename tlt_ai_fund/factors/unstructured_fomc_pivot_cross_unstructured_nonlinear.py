import numpy as np
import pandas as pd

class UnstructuredFomcPivotCrossFactor:
    """非结构化政策预期突变交叉因子 (unstructured/nonlinear)

    逻辑: 结合非结构化数据(FOMC情绪得分)与债券市场前瞻指标(2年期美债及收益率曲线)，捕捉美联储政策预期的极端跳跃。单纯的情绪得分属于低频阶梯数据，绝对禁止直接使用绝对值，必须使用5日边际变化来捕捉预期突变瞬间；同时要求短端利率(dgs2)同向剧烈重定价，以及收益率曲线(t10y2y)动量确认(牛陡/熊平)。三者非线性交叉形成高置信度的狙击手级脉冲信号。
    数据: fomc_sentiment, dgs2, t10y2y
    触发: fomc_sentiment 5日变化 Z-Score > 2.0 且 dgs2急降 且 t10y2y急剧变陡 -> 鸽派突变看多(+1.0)；反之极度鹰派且短端飙升看空(-1.0)
    输出: 脉冲型信号 [-1.0, 1.0]，非极端突变日休眠返回 0.0
    """

    def __init__(self, window: int = 126, fomc_z_thresh: float = 2.0, yield_z_thresh: float = 1.5):
        self.name = 'unstructured_fomc_pivot_cross'
        self.window = window
        self.fomc_z_thresh = fomc_z_thresh
        self.yield_z_thresh = yield_z_thresh

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始信号全为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['fomc_sentiment', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 提取并前向填充数据，避免缺失值导致计算中断
        df = data[required_cols].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only) 
        # 绝对禁止使用 fomc_sentiment 绝对值，使用5日变化代表一个完整交易周的动量，捕捉定价瞬间
        fomc_diff = df['fomc_sentiment'].diff(5)
        dgs2_diff = df['dgs2'].diff(5)
        t10y2y_diff = df['t10y2y'].diff(5)
        
        # 计算滚动 Z-Score (126个交易日约半年，作为近期波动的基准)
        # 增加极小值防止除以 0
        fomc_std = fomc_diff.rolling(self.window).std().replace(0.0, np.nan).fillna(1e-6)
        dgs2_std = dgs2_diff.rolling(self.window).std().replace(0.0, np.nan).fillna(1e-6)
        t10y2y_std = t10y2y_diff.rolling(self.window).std().replace(0.0, np.nan).fillna(1e-6)

        fomc_z = (fomc_diff - fomc_diff.rolling(self.window).mean()) / fomc_std
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(self.window).mean()) / dgs2_std
        t10y2y_z = (t10y2y_diff - t10y2y_diff.rolling(self.window).mean()) / t10y2y_std
        
        # 铁律2: 二阶导数/极端动量确认 + 非线性特征交叉
        # 鸽派突变 (Dovish Pivot): 情绪大幅转鸽(正向突变) + 短端利率暴跌(大幅下行) + 曲线急剧牛陡(利差大幅走阔)
        dovish_cond = (fomc_z > self.fomc_z_thresh) & (fomc_diff > 0) & \
                      (dgs2_z < -self.yield_z_thresh) & (dgs2_diff < 0) & \
                      (t10y2y_z > self.yield_z_thresh) & (t10y2y_diff > 0)
                      
        # 鹰派突变 (Hawkish Pivot): 情绪大幅转鹰(负向突变) + 短端利率飙升(大幅上行) + 曲线急剧熊平/倒挂(利差大幅收窄)
        hawkish_cond = (fomc_z < -self.fomc_z_thresh) & (fomc_diff < 0) & \
                       (dgs2_z > self.yield_z_thresh) & (dgs2_diff > 0) & \
                       (t10y2y_z < -self.yield_z_thresh) & (t10y2y_diff < 0)
        
        # 只在触发日赋予非零脉冲信号
        signal[dovish_cond] = 1.0
        signal[hawkish_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"UnstructuredFomcPivotCrossFactor(window={self.window}, fomc_z_thresh={self.fomc_z_thresh}, yield_z_thresh={self.yield_z_thresh})"