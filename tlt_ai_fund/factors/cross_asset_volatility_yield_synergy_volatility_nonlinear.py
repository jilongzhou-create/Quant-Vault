import numpy as np
import pandas as pd

class CrossAssetVolCurveFactor:
    """波动率极值与收益率曲线交叉反转因子 (volatility/nonlinear)

    逻辑: 恐慌极值叠加收益率曲线陡峭化是美债(TLT)的绝佳买点。从2022年的教训来看，单纯的VIX飙升或回落无法区分"通胀恐慌"(利空美债)与"衰退恐慌"(利多美债)。
          必须叠加 t10y2y 曲线动量: 
          1. 衰退恐慌/降息救市 (做多): 波动率极高且开始衰竭 + 收益率曲线急剧陡峭化 (短端下行确认降息预期)。
          2. 滞胀恐慌/激进加息 (做空): 波动率极高且继续升温 + 收益率曲线急剧平坦化 (短端上行确认紧缩预期)。
          3. 鹰派突袭 (做空): 极度自满被打破 + 曲线平坦化。
    数据: vixcls, gvzcls, t10y2y (或 t10y3m)
    触发: 波动率 126 日 Z-Score > 1.2 且 3 日动量衰竭, 同时期限利差 3 日陡峭化 > 2bps。
    输出: 严格的脉冲信号 [-1.0, 1.0]，常态休眠为 0.0。
    """

    def __init__(self):
        self.name = 'cross_asset_vol_curve'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 检查必要的基础数据
        if 'vixcls' not in data.columns:
            return signal
            
        # 获取收益率曲线数据 (优先 t10y2y，备用 t10y3m)
        if 't10y2y' in data.columns:
            spread = data['t10y2y'].ffill()
        elif 't10y3m' in data.columns:
            spread = data['t10y3m'].ffill()
        else:
            return signal

        vix = data['vixcls'].ffill()
        
        # 提取跨资产波动率 (黄金波动率，表征实际利率冲击)
        if 'gvzcls' in data.columns:
            gvz = data['gvzcls'].ffill()
            gvz_z = (gvz - gvz.rolling(126).mean()) / (gvz.rolling(126).std() + 1e-8)
            gvz_diff = gvz.diff(3)
        else:
            gvz_z = pd.Series(0.0, index=data.index)
            gvz_diff = pd.Series(0.0, index=data.index)

        # VIX Z-Score (126个交易日约为半年，捕捉宏观周期内的相对极值)
        vix_z = (vix - vix.rolling(126).mean()) / (vix.rolling(126).std() + 1e-8)

        # 铁律1 & 铁律2: 极值条件判定 (Z-Score绝对值 > 1.2 约覆盖顶部/底部10%-15%区间)
        panic_extreme = (vix_z > 1.2) | (gvz_z > 1.2)
        complacent_extreme = (vix_z < -1.2) | (gvz_z < -1.2)

        # 铁律3: 边际变化/二阶导数 (3日动量，捕捉趋势拐点)
        vix_diff = vix.diff(3)
        
        # 波动率边际降温 (衰竭)
        vol_cooling = (vix_diff < 0) | (gvz_diff < 0)
        # 波动率边际升温 (恶化)
        vol_heating = (vix_diff > 0) | (gvz_diff > 0)

        # 收益率曲线动量确认 (变陡=降息预期/衰退; 变平=加息预期/过热)
        curve_diff = spread.diff(3)
        curve_steepening = curve_diff > 0.02   # 3日内变陡 > 2bps
        curve_flattening = curve_diff < -0.02  # 3日内变平 < -2bps

        # 核心多空逻辑
        # 多头脉冲: 恐慌极值 + 恐慌边际衰竭 + 曲线陡峭化 = 确认降息救市，资金涌入美债
        long_cond = panic_extreme & vol_cooling & curve_steepening
        
        # 空头脉冲: 
        # 1. 极度自满 + 波动率边际抬头 + 曲线平坦化 = 鹰派突袭，美债承压
        # 2. 恐慌极值 + 恐慌持续升温 + 曲线平坦化 = 恶性滞胀/强力加息周期，美债遭抛售
        short_cond = (complacent_extreme & vol_heating & curve_flattening) | \
                     (panic_extreme & vol_heating & curve_flattening)

        # 赋值并清除可能的 NaN 误判
        signal[long_cond.fillna(False)] = 1.0
        signal[short_cond.fillna(False)] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"