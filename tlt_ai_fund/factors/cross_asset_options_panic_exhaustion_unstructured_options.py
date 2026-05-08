import numpy as np
import pandas as pd

class CrossAssetOptionsPanicExhaustionFactor:
    """跨资产期权恐慌衰竭因子 (unstructured/options)

    逻辑: VIX(股市恐慌)与GVZCLS(黄金恐慌)的隐含波动率差值，能够剥离出纯粹的“通缩式风险厌恶”(差值飙升)或“通胀式股市自满”(差值暴跌)。当通缩恐慌达到极值且开始衰竭时，通常是美联储转向降息救市的拐点，利好美债；当通胀恐慌极值且股市自满开始破裂时，通胀交易升温迫使美联储紧缩，利空美债。
    数据: vixcls, gvzcls
    触发: (VIX-GVZ)的 126日Z-Score > 1.5 且差值小于3日均值(恐慌衰竭) -> +1.0；Z-Score < -1.5 且差值大于3日均值(自满破裂) -> -1.0
    输出: 脉冲型信号，[-1.0, 1.0]，正值看多美债(TLT)，负值看空美债
    """

    def __init__(self):
        self.name = 'cross_asset_options_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号必须严格为 0.0 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需数据字段是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 前向填充缺失值
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产期权波动率差值
        vol_spread = vix - gvz
        
        # 计算126日(约半年)滚动均值和标准差，用于评估当前差值的极端程度
        roll_mean = vol_spread.rolling(window=126, min_periods=21).mean()
        roll_std = vol_spread.rolling(window=126, min_periods=21).std()
        
        # 避免除零错误
        zscore = (vol_spread - roll_mean) / roll_std.replace(0, np.nan)
        
        # 计算3日均值作为衰竭/拐点判断基准 (二阶导数铁律：必须等指标开始回落)
        short_ma = vol_spread.rolling(window=3, min_periods=1).mean()
        
        # 触发条件1：通缩恐慌极端（VIX极度高于GVZ）且开始衰竭（差值回落至3日均线以下）
        # 市场预期美联储即将下场救市，买入美债 (+1.0)
        long_cond = (zscore > 1.5) & (vol_spread < short_ma)
        
        # 触发条件2：通胀高压下的自满（GVZ极度高于VIX）且开始破裂（差值回升至3日均线以上）
        # 市场重新定价通胀风险及美联储紧缩，抛售美债 (-1.0)
        short_cond = (zscore < -1.5) & (vol_spread > short_ma)
        
        # 赋值脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"