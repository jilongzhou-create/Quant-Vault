import numpy as np
import pandas as pd

class VolCurveSteepeningReversalFactor:
    """波动率衰竭与曲线陡峭共振反转因子 (volatility/options)

    逻辑: 债市最强烈的趋势反转往往发生在"恐慌情绪极值消退 + 政策预期(宽松/紧缩)骤起"的瞬间。
          单看VIX极端飙升极易死于主跌浪接飞刀(如2022年加息周期)。因此必须严格遵守二阶导数铁律：
          等待VIX触及极端高位后开始回落衰竭(二阶导数<0), 同时伴随收益率曲线(10年-2年利差)
          发生剧烈的边际变陡(Bull Steepening脉冲, 确认降息预期), 两者共振才产生做多美债(TLT)信号。
          反之, 当波动率极度死寂被打破且伴随曲线剧烈平坦化(Bear Flattening脉冲)时, 视为紧缩恐慌开启的看空脉冲。
          因子严格遵守零值休眠, 仅在这些边缘变化共振的极少数日子输出信号。
    数据: vixcls (VIX期权波动率), t10y2y (10年-2年期限利差)
    触发: 
      多头脉冲: VIX 126日 Z-Score > 1.5 且开始衰竭回落 + t10y2y 日变动大于过去63日动量波动的1.0倍。
      空头脉冲: VIX 126日 Z-Score < -1.0 且开始抬头飙升 + t10y2y 日变动小于负的过去63日动量波动的1.0倍。
    输出: +1.0 (看多美债) / -1.0 (看空美债) / 0.0 (休眠)
    """

    def __init__(self):
        self.name = 'vol_curve_steepening_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠, 初始全为0.0
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['vixcls', 't10y2y']
        if not set(req_cols).issubset(data.columns):
            return signal
            
        vix = data['vixcls'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # --- 模块1: 波动率极值与衰竭的微观结构 ---
        # 计算 126日 (半年) 的 VIX 偏离度
        vix_mean = vix.rolling(window=126).mean()
        vix_std = vix.rolling(window=126).std()
        vix_z = (vix - vix_mean) / vix_std
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife), 测量衰竭和反弹
        vix_3d_mean = vix.rolling(window=3).mean()
        vix_diff = vix.diff()
        
        # --- 模块2: 收益率曲线预期的边际突变脉冲 ---
        # 铁律3: 边际变化, 摒弃绝对水位, 仅看日内利差动量
        t10y2y_diff = t10y2y.diff()
        t10y2y_diff_std = t10y2y_diff.rolling(window=63).std()
        
        # --- 模块3: 跨资产共振信号判定 ---
        
        # 做多脉冲: 恐慌极值开始消退 (VIX高位回落) + 降息预期脉冲骤起 (曲线大幅陡峭化)
        vix_panic_exhaustion = (vix_z > 1.5) & (vix < vix_3d_mean) & (vix_diff < 0)
        bull_steepening_pulse = t10y2y_diff > (1.0 * t10y2y_diff_std)
        long_cond = vix_panic_exhaustion & bull_steepening_pulse
        
        # 做空脉冲: 极度贪婪打破 (VIX低迷抬头) + 紧缩预期脉冲骤起 (曲线大幅平坦化/倒挂加深)
        vix_greed_breakout = (vix_z < -1.0) & (vix > vix_3d_mean) & (vix_diff > 0)
        bear_flattening_pulse = t10y2y_diff < (-1.0 * t10y2y_diff_std)
        short_cond = vix_greed_breakout & bear_flattening_pulse
        
        # 信号赋值
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        # 数据平滑早期的缺失值和极值产生的非法结果清洗
        signal = signal.fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"