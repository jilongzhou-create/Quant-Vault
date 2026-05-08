import numpy as np
import pandas as pd

class UnstructuredFomcCurvePivotFactor:
    """FOMC 预期突变与曲线验证脉冲 (Unstructured/Unstructured)

    逻辑: 捕捉 FOMC 声明情绪的罕见边际反转，要求市场收益率曲线(短端利率+期限利差)给出形态印证，并在趋势动能出现初步衰竭(二阶导数)时发出狙击脉冲。这种动量确立加短期衰竭的结构能有效避免单边主跌浪。
    数据: fomc_sentiment, dgs2, t10y2y
    触发: FOMC情绪跳跃Z-Score极值 + dgs2/t10y2y趋势印证 + dgs2单日动能衰竭。
    输出: 脉冲信号 +1.0 看多美债(政策转鸽+牛陡), -1.0 看空美债(政策转鹰+熊平), 常态 0.0。
    """

    def __init__(self, z_window=252, diff_window=3):
        self.name = 'unstructured_fomc_curve_pivot'
        self.z_window = z_window
        self.diff_window = diff_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 数据存在性校验
        req_cols = ['fomc_sentiment', 'dgs2', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        fomc = data['fomc_sentiment'].ffill()
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # ==========================================
        # 铁律3: 边际变化 (严禁使用阶梯数据的绝对值)
        # ==========================================
        fomc_diff = fomc.diff(self.diff_window)
        
        # 计算 FOMC 情绪突变的 Z-Score
        # 施加 .clip(lower=0.01) 避免长期的无会议空窗期导致方差趋于0，进而引发假突破信号
        fomc_std = fomc_diff.rolling(self.z_window).std().replace(0, np.nan).ffill().clip(lower=0.01)
        fomc_mean = fomc_diff.rolling(self.z_window).mean()
        fomc_z = (fomc_diff - fomc_mean) / fomc_std
        
        # 计算市场端现货的边际响应
        dgs2_diff = dgs2.diff(self.diff_window)
        t10y2y_diff = t10y2y.diff(self.diff_window)
        
        # ==========================================
        # 铁律2: 二阶导数 (必须等待极速动能衰竭才能入场，防接飞刀)
        # ==========================================
        # 对比当日变动与前段窗口平均动量，只有当最新变动放缓时，认定势能进入第一波衰竭期
        dgs2_daily_diff = dgs2.diff(1)
        dgs2_prev_momentum = dgs2_diff.shift(1) / self.diff_window
        
        # --- 看多脉冲 (Bullish Pulse) ---
        # 1. FOMC NLP情绪突发转鸽 (鸽派突变 Z-score > 2.5)
        # 2. 市场验证降息预期爆发 (dgs2 快速下行超 5bp) 且 曲线呈现牛陡形态 (Bull Steepening, 利差扩阔超 2bp)
        # 3. 二阶衰竭验证: dgs2的当日下行不再强于之前的平均速度 (跌势企稳)
        bullish_fomc = fomc_z > 2.5
        bullish_market = (dgs2_diff < -0.05) & (t10y2y_diff > 0.02)
        bullish_exhaustion = dgs2_daily_diff > dgs2_prev_momentum 
        
        bullish_pulse = bullish_fomc & bullish_market & bullish_exhaustion
        
        # --- 看空脉冲 (Bearish Pulse) ---
        # 1. FOMC NLP情绪突发转鹰 (鹰派突变 Z-score < -2.5)
        # 2. 市场验证加息预期发酵 (dgs2 快速上行超 5bp) 且 曲线呈现熊平形态 (Bear Flattening, 利差收窄超 2bp)
        # 3. 二阶衰竭验证: dgs2的当日上行不再强于之前的平均速度 (涨抛势能释放完毕)
        bearish_fomc = fomc_z < -2.5
        bearish_market = (dgs2_diff > 0.05) & (t10y2y_diff < -0.02)
        bearish_exhaustion = dgs2_daily_diff < dgs2_prev_momentum
        
        bearish_pulse = bearish_fomc & bearish_market & bearish_exhaustion
        
        # ==========================================
        # 铁律1: 零值休眠 (非极端触发日必须返回 0.0)
        # ==========================================
        signal[bullish_pulse] = 1.0
        signal[bearish_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_window={self.z_window}, diff_window={self.diff_window})"