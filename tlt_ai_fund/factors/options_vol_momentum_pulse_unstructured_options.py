import numpy as np
import pandas as pd

class OptionsVolMomentumPulseFactor:
    """期权波动率动量衰竭脉冲因子 (unstructured/options)

    逻辑: 捕捉期权隐含波动率(VIX)的短期剧烈边际变化(动量冲击)。当波动率在5天内出现极端飙升(反映期权市场突发性恐慌定价)时, 美债往往面临流动性无差别抛售; 只有当波动率边际回落(恐慌衰竭)的瞬间, 避险资金才真正流入美债, 触发看多脉冲。反之, 波动率暴跌后企稳反弹, 表明极度Risk-On情绪衰竭, 触发看空脉冲。
    数据: vixcls (CBOE VIX期权隐含波动率)
    触发: 
      - 看多: VIX 5日变化量的 126日 Z-Score > 2.0 (极端恐慌飙升) 且 单日 diff() < 0 (开始回落/衰竭)
      - 看空: VIX 5日变化量的 126日 Z-Score < -2.0 (极端自满暴跌) 且 单日 diff() > 0 (企稳反弹)
    输出: 严格脉冲型, +1.0 看多美债, -1.0 看空美债, 常态为 0.0
    """

    def __init__(self, momentum_days=5, window=126, z_thresh=2.0):
        self.name = 'options_vol_momentum_pulse'
        self.momentum_days = momentum_days
        self.window = window
        self.z_thresh = z_thresh

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 信号 (铁律1: 零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns:
            return signal

        vix = data['vixcls'].ffill()
        
        # 铁律3: 边际变化 - 绝对禁止使用 VIX 绝对水位, 必须计算动量变化
        vix_momentum = vix.diff(self.momentum_days)
        
        # 计算基于动量的滚动 Z-Score (使用 126 个交易日即半年的滚动窗口, 适应不同波动率中枢)
        roll_mean = vix_momentum.rolling(window=self.window, min_periods=self.window // 2).mean()
        roll_std = vix_momentum.rolling(window=self.window, min_periods=self.window // 2).std()
        z_score = (vix_momentum - roll_mean) / (roll_std + 1e-8)
        
        # 铁律2: 二阶导数 - 单日边际变化开始衰竭 (防止接飞刀死于主跌浪)
        daily_diff = vix.diff(1)
        exhaustion_long = daily_diff < 0   # 飙升后的第一天回落
        exhaustion_short = daily_diff > 0  # 暴跌后的第一天反弹
        
        # 组合脉冲触发条件
        long_trigger = (z_score > self.z_thresh) & exhaustion_long
        short_trigger = (z_score < -self.z_thresh) & exhaustion_short
        
        # 赋值信号
        signal[long_trigger] = 1.0
        signal[short_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"OptionsVolMomentumPulseFactor(mom_days={self.momentum_days}, win={self.window}, z={self.z_thresh})"