import numpy as np
import pandas as pd

class RealYieldPivotFactor:
    """实际利率政策转向脉冲 (policy_pivot/nonlinear)

    逻辑: 捕捉美联储超预期放松/收紧导致的实际利率突变。实际利率(TIPS)急跌且通胀预期(Breakeven)未崩盘，代表纯粹的鸽派流动性释放(非衰退恐慌)，强烈看多美股；反之实际利率急升且通胀预期未飙升，代表纯粹的鹰派收水，看空美股。
    数据: [dfii5, t5yie]
    输出: 鸽派流动性宽松脉冲为+1.0，鹰派收缩脉冲为-1.0
    触发条件: 5年期实际利率5日内急剧变化绝对值>15bp且Z-Score>1.5，叠加通胀预期非极端反向，预期Trigger Rate约 5%-10%
    """

    def __init__(self, lookback=5, z_window=252, real_bp_threshold=0.15, be_bp_threshold=0.05, z_threshold=1.5):
        self.name = 'real_yield_pivot_pulse'
        self.lookback = lookback
        self.z_window = z_window
        self.real_bp_threshold = real_bp_threshold
        self.be_bp_threshold = be_bp_threshold
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 默认输出全 0.0 的脉冲信号序列
        signal = pd.Series(0.0, index=data.index)
        
        # 检查数据完备性
        if 'dfii5' not in data.columns or 't5yie' not in data.columns:
            return signal
            
        dfii5 = data['dfii5'].ffill()
        t5yie = data['t5yie'].ffill()
        
        # 计算动量变化 (严格遵守边际变化铁律)
        dfii5_diff = dfii5.diff(self.lookback)
        t5yie_diff = t5yie.diff(self.lookback)
        
        # 计算实际利率变化的 Z-Score (识别历史极值状态)
        dfii5_diff_mean = dfii5_diff.rolling(window=self.z_window, min_periods=self.lookback).mean()
        dfii5_diff_std = dfii5_diff.rolling(window=self.z_window, min_periods=self.lookback).std()
        dfii5_z = (dfii5_diff - dfii5_diff_mean) / (dfii5_diff_std + 1e-8)
        
        # 今日动量方向确认 (确保转向趋势正在发生)
        dfii5_daily = dfii5.diff(1)
        
        # 鸽派流动性突变 (买入信号 +1.0)
        # 1. 实际利率暴跌 (达到绝对阈值15bp + 历史1.5倍标准差极值)
        # 2. 盈亏平衡通胀预期没有暴跌 (过滤掉通缩/衰退恐慌导致的利率下行，确保是央行主动降息的软着陆剧本)
        # 3. 当日实际利率仍在下行或企稳
        long_cond = (
            (dfii5_diff <= -self.real_bp_threshold) & 
            (dfii5_z <= -self.z_threshold) & 
            (t5yie_diff >= -self.be_bp_threshold) & 
            (dfii5_daily <= 0)
        )
        
        # 鹰派流动性收紧 (卖出信号 -1.0)
        # 1. 实际利率暴涨 (市场抢跑加息/推迟降息)
        # 2. 通胀预期没有暴涨 (纯粹的流动性收缩，而非经济过热主导)
        # 3. 当日实际利率仍在上升或企稳
        short_cond = (
            (dfii5_diff >= self.real_bp_threshold) & 
            (dfii5_z >= self.z_threshold) & 
            (t5yie_diff <= self.be_bp_threshold) & 
            (dfii5_daily >= 0)
        )
        
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback={self.lookback}, real_bp={self.real_bp_threshold})"