import numpy as np
import pandas as pd

class BullSteepeningLiquidityPulseFactor:
    """Bull Steepening 与流动性重定价脉冲因子 (policy_pivot/nonlinear)

    逻辑: 捕捉短端利率急剧下行导致美债曲线“牛陡”(Bull Steepening)的瞬间。当2年期国债收益率短期暴跌、期限利差陡升，且伴随VIX恐慌情绪衰竭时，标志着政策转向(降息抢跑)被市场实质确认，风险偏好修复，触发看多脉冲；反之，若短端利率飙升导致“熊平”且伴随恐慌上升，则触发紧缩看空脉冲。
    数据: [dgs2, t10y2y, vixcls]
    输出: [-1.0, 1.0] 的脉冲信号。正值看多(流动性宽松突变)，负值看空(紧缩预期突增)
    触发条件: 2年期美债收益率5日降幅超12bps 且 T10Y2Y变陡超8bps 且 VIX近期回落。预期Trigger Rate约 5%~12%。
    """

    def __init__(self, rate_window: int = 5, vix_window: int = 3, dgs2_thresh: float = 0.12, steepening_thresh: float = 0.08):
        self.name = 'bull_steepening_liquidity_pulse'
        self.rate_window = rate_window
        self.vix_window = vix_window
        self.dgs2_thresh = dgs2_thresh  # 12个基点，代表急剧重定价的幅度
        self.steepening_thresh = steepening_thresh  # 8个基点，代表曲线显著变陡

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['dgs2', 't10y2y', 'vixcls']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        # 针对宏观低频或节假日导致的数据空缺进行向前填充
        df = data[required_cols].ffill()
        
        # 核心物理法则：计算边际动量变化，禁止使用绝对值
        dgs2_diff = df['dgs2'].diff(self.rate_window)
        t10y2y_diff = df['t10y2y'].diff(self.rate_window)
        vix_diff = df['vixcls'].diff(self.vix_window)
        
        # 脉冲触发条件 1：政策转向看多 (牛陡突变 + 恐慌衰竭防飞刀)
        bull_cond = (dgs2_diff <= -self.dgs2_thresh) & (t10y2y_diff >= self.steepening_thresh) & (vix_diff <= 0.0)
        
        # 脉冲触发条件 2：紧缩冲击看空 (熊平突变 + 恐慌发酵)
        bear_cond = (dgs2_diff >= self.dgs2_thresh) & (t10y2y_diff <= -self.steepening_thresh) & (vix_diff >= 0.0)
        
        signal[bull_cond] = 1.0
        signal[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(rate_window={self.rate_window}, vix_window={self.vix_window})"