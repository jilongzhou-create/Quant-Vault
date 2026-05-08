import numpy as np
import pandas as pd

class RealYieldSoftLandingPulseFactor:
    """真实收益率软着陆确认脉冲因子 (policy_pivot/nonlinear)

    逻辑: 实际收益率(DFII5)是美联储流动性最直观的定价锚。其剧烈下行代表货币政策预期出现显著转鸽。此时若高收益债信用利差(HYM2)同步收窄, 说明信用市场将此降息预期定性为'软着陆'而非衰退恐慌(排除硬着陆), 此时产生强烈做多脉冲; 反之为鹰派紧缩引发的杀估值冲击。
    数据: dfii5(5年期实际收益率), bamlh0a0hym2(高收益债利差), vixcls(VIX指数)
    输出: 1.0(流动性转宽且软着陆确认), -1.0(流动性收紧且风险偏好恶化)
    触发条件: 实际收益率5日动量Z-Score极值共振利差收窄及恐慌回落, 预期Trigger Rate约5%-10%
    """

    def __init__(self):
        self.name = 'real_yield_soft_landing_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全零信号 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['dfii5', 'bamlh0a0hym2', 'vixcls']
        if not all(col in data.columns for col in req_cols):
            return signal
            
        df = data[req_cols].ffill()
        
        # 1. 实际收益率5日动量及其1年(252日)滚动Z-Score (衡量宽松/紧缩的瞬时冲击)
        # 使用5日来平滑单日杂音，捕捉一周级别的定价巨变
        dfii5_diff = df['dfii5'].diff(5)
        dfii5_roll_mean = dfii5_diff.rolling(window=252, min_periods=126).mean()
        dfii5_roll_std = dfii5_diff.rolling(window=252, min_periods=126).std()
        dfii5_z = (dfii5_diff - dfii5_roll_mean) / dfii5_roll_std
        
        # 2. 高收益债信用利差3日动量及其Z-Score (衰退确认器)
        # 若利差大幅走阔说明是由于经济衰退而抢跑降息；若利差收窄说明是预防式降息（软着陆）
        hy_diff = df['bamlh0a0hym2'].diff(3)
        hy_roll_mean = hy_diff.rolling(window=252, min_periods=126).mean()
        hy_roll_std = hy_diff.rolling(window=252, min_periods=126).std()
        hy_z = (hy_diff - hy_roll_mean) / hy_roll_std
        
        # 3. 恐慌情绪边际变化 (防接飞刀二阶导数铁律)
        vix_diff = df['vixcls'].diff(1)
        
        # 鸽派软着陆脉冲 (+1.0)
        # 条件: 实际收益率极速暴跌(Z < -1.25) + 信用利差收窄(排除衰退风险) + 恐慌边际消退(VIX下降)
        bull_cond = (dfii5_z < -1.25) & (hy_z < -0.5) & (vix_diff < 0)
        
        # 鹰派紧缩冲击脉冲 (-1.0)
        # 条件: 实际收益率极速飙升(Z > 1.25) + 信用利差走阔(流动性枯竭/违约风险) + 恐慌边际升温(VIX上升)
        bear_cond = (dfii5_z > 1.25) & (hy_z > 0.5) & (vix_diff > 0)
        
        signal[bull_cond] = 1.0
        signal[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"