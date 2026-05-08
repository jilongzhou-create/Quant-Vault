import numpy as np
import pandas as pd

class VolatilityCurveRegimeFactor:
    """波动率与曲线结构极值因子 (volatility/options)

    逻辑: 纯粹的 VIX 极值反转胜率极低, 因为在通胀恐慌(2022)和通缩恐慌(2020)中债市表现完全相反。本因子将期权隐含波动率(VIX)的脉冲衰竭与收益率曲线(t10y2y)的边际动量结合。当 VIX 极度恐慌且开始回落时，若曲线陡峭化(联储降息预期起效)，代表通缩恐慌修复，强烈看多美债；若曲线平坦化(联储仍需加息压制)，则是通胀主导的股债双杀，看空美债。严格满足零值休眠和二阶导数铁律。
    数据: vixcls (波动率), t10y2y (收益率曲线利差)
    触发: VIX 252日 Z-Score > 2.0 且低于5日均线 (极值+衰竭), 配合 t10y2y 的 5 日动量决定方向; 或 VIX Z-Score < -1.5 且向上突破配合曲线决定方向。
    输出: +1.0 看多美债, -1.0 看空美债, 非触发期严格 0.0
    """

    def __init__(self):
        self.name = 'vol_curve_regime_options_macro'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 异常数据处理: 缺少所需列直接返回全 0 Series
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            return pd.Series(0.0, index=data.index)

        vix = data['vixcls'].ffill()
        curve = data['t10y2y'].ffill()

        # 铁律1 & 2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 计算 252 日 Z-Score, 捕捉年度级别的宏观极值脉冲
        vix_rolling_mean = vix.rolling(252).mean()
        vix_rolling_std = vix.rolling(252).std()
        vix_z252 = (vix - vix_rolling_mean) / vix_rolling_std
        
        # 衰竭条件: 极值发生后, 必须跌破或突破 5 日均线才触发
        vix_ma5 = vix.rolling(5).mean()
        vix_extreme_high = vix_z252 > 2.0
        vix_exhaustion = vix < vix_ma5
        
        vix_extreme_low = vix_z252 < -1.5
        vix_surging = vix > vix_ma5

        # 铁律3: 边际变化 (Marginal Change Only)
        # 观察短端利率与长端利率预期的边际博弈变化 (5日动量)
        curve_diff = curve.diff(5)

        # 初始化输出信号 (零值休眠)
        signal = pd.Series(0.0, index=data.index)

        # 脉冲触发条件:
        # 多头信号 (+1.0): 避险情绪缓解/恐慌发酵 + 联储降息预期强烈(曲线陡峭化)
        long_cond1 = vix_extreme_high & vix_exhaustion & (curve_diff > 0.02)
        long_cond2 = vix_extreme_low & vix_surging & (curve_diff > 0.02)
        
        # 空头信号 (-1.0): 恐慌缓解/安逸被打破 + 联储紧缩预期强烈(曲线平坦化,如2022年)
        short_cond1 = vix_extreme_high & vix_exhaustion & (curve_diff < -0.02)
        short_cond2 = vix_extreme_low & vix_surging & (curve_diff < -0.02)

        # 赋值并保证互斥
        signal[long_cond1 | long_cond2] = 1.0
        signal[short_cond1 | short_cond2] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"