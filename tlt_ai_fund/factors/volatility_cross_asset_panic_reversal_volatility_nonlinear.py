import numpy as np
import pandas as pd

class VolatilityCrowdingReversalFactor:
    """Volatility Crowding Reversal (volatility/nonlinear)

    逻辑: 跨资产波动率(美债短端利率、VIX、黄金)在极端狂飙后同步衰竭时，标志着宏观恐慌情绪见顶。此时通过观察短端利率(dgs2，即联储政策预期的直接 proxy)的边际动量来确认资金的真实流向，顺势捕捉趋势恢复的脉冲机会，避免在波动率极值点接飞刀。
    数据: dgs2 (美债2年期收益率), vixcls (VIX指数), gvzcls (黄金VIX)
    触发: 任一宏观波动率的 252日 Z-Score > 1.5 (满足极值)，且至少两项波动率低于3日均值 (满足衰竭与跨资产确认)，同时顺应短端利率的21日边际动量。
    输出: +1.0 (恐慌衰竭且利率下行，看多美债) / -1.0 (恐慌衰竭且利率上行，看空美债) / 0.0 (常态休眠)
    """

    def __init__(self):
        self.name = 'volatility_crowding_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必需的字段是否存在 (避免使用 CoreAnchor 数据)
        required_cols = ['dgs2', 'vixcls', 'gvzcls']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 提取并前向填充数据
        dgs2 = data['dgs2'].ffill()
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 1. 提取利率市场自身的波动率特征 (美债短端脉冲)
        dgs2_diff = dgs2.diff()
        dgs2_vol = dgs2_diff.rolling(window=21).std()
        
        # 2. 计算各资产波动率的 Z-Score (长期锚定 252个交易日)
        def calc_zscore(s: pd.Series, window: int = 252) -> pd.Series:
            return (s - s.rolling(window).mean()) / s.rolling(window).std()
            
        vol_z = calc_zscore(dgs2_vol)
        vix_z = calc_zscore(vix)
        gvz_z = calc_zscore(gvz)
        
        # 3. 铁律2 & 零值休眠: 识别跨资产极端恐慌
        # (使用 1.5倍标准差以保证 5%~15% 的 Trigger Rate 要求)
        is_extreme = (vol_z > 1.5) | (vix_z > 1.5) | (gvz_z > 1.5)
        
        # 4. 铁律2: 二阶导数衰竭确认 (Anti-Catch-Falling-Knife)
        # 当前波动率低于短期(3日)均值，且要求不同资产类别间存在共振衰竭 (至少两项达成)
        vol_exhaust = dgs2_vol < dgs2_vol.rolling(window=3).mean()
        vix_exhaust = vix < vix.rolling(window=3).mean()
        gvz_exhaust = gvz < gvz.rolling(window=3).mean()
        
        exhaust_count = vol_exhaust.astype(int) + vix_exhaust.astype(int) + gvz_exhaust.astype(int)
        is_exhausting = exhaust_count >= 2
        
        # 5. 铁律3: 边际变化与高胜率方向确认 (Hit Rate > 55% 的核心)
        # 恐慌消退后，美债的真实方向取决于政策利率的边际 repricing 趋势
        mom5 = dgs2.diff(periods=5)
        mom21 = dgs2.diff(periods=21)
        
        # 短端利率下行幅度 > 5bps (联储转鸽，美债走牛)
        bull_cond = (mom5 < 0) & (mom21 < -0.05)
        # 短端利率上行幅度 > 5bps (联储维持鹰派，美债走熊)
        bear_cond = (mom5 > 0) & (mom21 > 0.05)
        
        # 6. 生成狙击手级别脉冲信号
        trigger = is_extreme & is_exhausting
        
        signal[trigger & bull_cond] = 1.0
        signal[trigger & bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"