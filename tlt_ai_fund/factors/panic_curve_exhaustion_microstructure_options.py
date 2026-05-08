import numpy as np
import pandas as pd

class PanicCurveExhaustionFactor:
    """微观结构/期权波动率 (Panic Curve Exhaustion)

    逻辑: 结合美股隐含波动率(VIX)极值与美债收益率曲线(10Y-2Y)的边际动量。在极端流动性恐慌爆发初期，现金为王常导致美债被无差别抛售；只有当恐慌见顶衰竭(VIX触及极值后二阶回落)，且债市给出右侧确认(收益率曲线边际变陡，说明市场计入宽松/救市预期)时，才是高胜率的抄底脉冲。反之，在紧缩末期曲线极度倒挂时，若倒挂进一步加深且股市波动率抬头，则触发看空脉冲规避股债双杀。常态严格零值休眠。
    数据: vixcls, t10y2y
    触发: 多头 -> VIX Z-Score > 1.5 AND VIX < 3日均值 AND T10Y2Y.diff(3) > 0; 空头 -> T10Y2Y Z-Score < -1.5 AND T10Y2Y.diff(3) < 0 AND VIX > 3日均值
    输出: [-1.0, 1.0] 的狙击手脉冲信号
    """

    def __init__(self):
        self.name = 'microstructure_panic_curve_exhaustion_options'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 必须验证数据列是否存在，处理缺失情况
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            signal.name = self.name
            return signal
            
        vix = data['vixcls'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 1. 长周期极端水位识别 (252日 Z-Score，反映年内极端情绪)
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        vix_zscore = (vix - vix_mean) / vix_std
        
        t10_mean = t10y2y.rolling(window=252, min_periods=60).mean()
        t10_std = t10y2y.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        t10_zscore = (t10y2y - t10_mean) / t10_std
        
        # 2. 衰竭与边际变化确认 (严格执行二阶导数 & 边际变化铁律)
        # 恐慌极值开始衰退 (VIX 失去上升动能，向下拐头)
        vix_falling = vix < vix.rolling(window=3).mean()
        # 股市压力开始上升 (VIX 向上拐头)
        vix_rising = vix > vix.rolling(window=3).mean()
        
        # 收益率曲线动量: 变陡(Bull Steepening / Bear Steepening) 标志宽松起效或风险溢价回归
        curve_steepening = t10y2y.diff(3) > 0
        # 收益率曲线动量: 平坦化/倒挂加深(Bear Flattening) 标志短期紧缩超预期发酵
        curve_flattening = t10y2y.diff(3) < 0
        
        # 3. 信号组合 (Sniper Pulse 铁律)
        # 看多脉冲 (+1.0): 流动性恐慌见顶衰竭 + 收益率曲线确认变陡 (预示避险资金流入及货币政策可能转向)
        long_cond = (vix_zscore > 1.5) & vix_falling & curve_steepening
        
        # 看空脉冲 (-1.0): 紧缩扭曲极点 + 短期紧缩继续破位倒挂 + 股市开始感受到紧缩痛楚
        short_cond = (t10_zscore < -1.5) & curve_flattening & vix_rising
        
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"