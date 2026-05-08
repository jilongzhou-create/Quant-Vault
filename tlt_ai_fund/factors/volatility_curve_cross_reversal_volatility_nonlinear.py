import numpy as np
import pandas as pd

class VolatilityCurveCrossReversalFactor:
    """波动率与曲线动量交叉反转因子 (volatility/nonlinear)

    逻辑: 结合跨资产波动率极值与收益率曲线动量的高维交叉因子。当股市处于极端恐慌(VIX高位)且开始回落衰竭时，如果伴随收益率曲线在短时间内急剧变陡(短端利率大幅下行，反映宏观资金紧急定价美联储宽松降息)，则共振确认为避险多头反转脉冲。反之，当VIX处于极端自满低位且开始触底抬头，伴随曲线快速平坦化(反映紧缩加息预期升温)，则触发做空脉冲。因子严格遵守零值休眠、二阶衰竭及边际变化三大铁律。
    数据: vixcls (VIX波动率), t10y2y (收益率曲线利差)
    触发: VIX 126日 Z-Score > 2.0 且跌破3日均值(恐慌衰竭) + t10y2y 5日动量 > 5bps(曲线边际陡峭化) -> +1.0；空头反之。
    输出: 狙击手型脉冲信号，非触发日严格保持 0.0，触发时输出 +1.0 或 -1.0。
    """

    def __init__(self):
        self.name = 'vol_curve_cross_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖字段是否存在 (处理数据缺失铁律)
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        # 缺失值前向填充
        vix = data['vixcls'].ffill()
        curve = data['t10y2y'].ffill()
        
        # 1. 极值条件: 动态计算VIX的126日(半年)滚动Z-Score, 识别极端恐慌/极端自满
        vix_mean = vix.rolling(window=126).mean()
        vix_std = vix.rolling(window=126).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-6)
        
        # 2. 二阶导数(Anti-Catch-Falling-Knife): 
        # 绝对禁止VIX狂飙时接飞刀，必须等VIX跌破3日均值确认恐慌动能衰竭
        # 做空同理，必须等极度自满被打破，VIX升破3日均值
        vix_exhaustion_long = vix < vix.rolling(window=3).mean()
        vix_exhaustion_short = vix > vix.rolling(window=3).mean()
        
        # 3. 边际变化(Marginal Change): 
        # 摒弃曲线绝对倒挂水位，仅关注短时间内的动量突变
        # 5日陡峭化 > 5个基点(0.05) 代表降息预期急剧爆发 (Bull Steepening)
        # 5日平坦化 < -5个基点(-0.05) 代表紧缩预期急剧升温 (Bear Flattening)
        curve_mom = curve.diff(5)
        curve_steepening = curve_mom > 0.05
        curve_flattening = curve_mom < -0.05
        
        # 4. 高维非线性交叉触发
        long_cond = (vix_z > 2.0) & vix_exhaustion_long & curve_steepening
        short_cond = (vix_z < -1.5) & vix_exhaustion_short & curve_flattening
        
        # 零值休眠铁律: 常态 0.0, 仅满足苛刻条件时输出脉冲
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"