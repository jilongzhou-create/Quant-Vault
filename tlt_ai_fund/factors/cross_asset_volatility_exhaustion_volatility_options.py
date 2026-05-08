import numpy as np
import pandas as pd

class VolCurveMicrostructureFactor:
    """Volatility and Yield Curve Microstructure Reversal

    逻辑: 捕捉极端恐慌和极度贪婪的边际反转，并通过收益率曲线动量区分"通胀恐慌"与"衰退恐慌"。
    单纯做多波动率衰竭会死于通胀加息周期(如2022年，此时VIX回落往往伴随美债暴跌)。
    因此必须引入二阶导数与曲线确认：当VIX处于极值并边际反转时，如果收益率曲线边际变陡(短端相对下行)，
    说明核心矛盾是衰退或美联储转向，此时避险与宽松预期共振，强烈看多美债(+1.0)；
    如果曲线边际走平(紧缩继续)，说明核心矛盾仍是通胀，波动率事件只是风险资产的流动性冲击，坚决看空美债(-1.0)。
    数据: vixcls (波动率), t10y2y (期限利差)
    触发: VIX 126日 Z-Score 绝对值 > 1.25 且 突破3日均线(衰竭/飙升)，配合 t10y2y 的3日边际动量方向。
    输出: +1.0 (做多美债) 或 -1.0 (做空美债) 的狙击手级脉冲信号。
    """
    
    def __init__(self):
        self.name = 'vol_curve_micro_reversal'
        
    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 - 常态下必须输出 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        curve = data['t10y2y'].ffill()
        
        # 计算 VIX 的中短期宏观极值 (126日 = 半年)
        vix_mean = vix.rolling(window=126, min_periods=60).mean()
        vix_std = vix.rolling(window=126, min_periods=60).std()
        vix_std = vix_std.replace(0, np.nan)  # 防止除以零
        vix_z = (vix - vix_mean) / vix_std
        
        # 铁律2: 二阶导数 - 绝对禁止"波动率高=买入"，必须叠加边际回落/飙升
        vix_ma3 = vix.rolling(window=3).mean()
        
        # 恐慌极值 + 开始衰竭 (Volatility Exhaustion)
        vix_exhaustion = (vix_z > 1.25) & (vix < vix_ma3)
        # 极度贪婪 + 恐慌初现 (Volatility Surge)
        vix_surge = (vix_z < -1.25) & (vix > vix_ma3)
        
        # 整合所有波动率极值突变事件
        vol_event = vix_exhaustion | vix_surge
        
        # 铁律3: 边际变化 - 利用收益率曲线的3日动量判断宏观流动性环境
        curve_mom = curve.diff(3)
        
        # 曲线陡峭化: 短端利率下行预期，通常对应衰退交易或宽松预期
        curve_steepening = curve_mom > 0.0
        # 曲线平坦化: 短端坚挺或继续加息，通常对应通胀交易或紧缩预期
        curve_flattening = curve_mom < 0.0
        
        # 生成狙击手脉冲信号
        bull_trigger = vol_event & curve_steepening
        bear_trigger = vol_event & curve_flattening
        
        # 赋值触发日信号
        signal[bull_trigger] = 1.0
        signal[bear_trigger] = -1.0
        
        # 清理异常逻辑冲突日
        conflict = bull_trigger & bear_trigger
        signal[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(Z-Score Threshold=1.25, MA=3, Momentum=3)"