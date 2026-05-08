import numpy as np
import pandas as pd

class CrossAssetVolPanicReversalFactor:
    """跨资产极值恐慌反转因子 (volatility/options)

    逻辑: 监控 VIX(美股) 和 GVZCLS(黄金) 跨资产期权隐含波动率的系统性恐慌。
          当宏观流动性危机爆发时(Dash for cash), 股债金往往遭到无差别抛售，导致跨资产波动率同步极端飙升(如2020年3月)。
          基于二阶导数铁律，绝对不能在波动率冲顶时买入美债。
          一旦极端波动率见顶并开始双双回落(二阶导数衰竭), 标志着无差别抛售结束，
          市场理智回归，避险资金将迅速重返终极安全资产美债(TLT)。这是一个典型的狙击手级脉冲信号。
    数据: vixcls (VIX指数), gvzcls (黄金隐含波动率)
    触发: VIX 252日 Z-Score > 2.5 且 VIX 向下击穿3日均线(动量衰竭) 且 GVZCLS 也击穿3日均线(跨资产确认)
    输出: +1.0 (脉冲式看多美债)
    """

    def __init__(self):
        self.name = 'options_cross_asset_vol_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 检查所需数据列是否存在
        required_cols = ['vixcls', 'gvzcls']
        if not all(col in data.columns for col in required_cols):
            return signal

        # 获取数据并向前填充处理缺失值
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()

        # 1. 极值条件: 计算 VIX 252个交易日(一年)的 Z-Score
        vix_rolling_mean = vix.rolling(window=252, min_periods=120).mean()
        vix_rolling_std = vix.rolling(window=252, min_periods=120).std()
        
        # 防止标准差为0导致的除零错误
        vix_rolling_std = vix_rolling_std.replace(0, np.nan)
        vix_zscore = (vix - vix_rolling_mean) / vix_rolling_std

        # 2. 衰竭条件 (二阶导数铁律): 波动率不能仅仅是高，必须已经打破极短期的上涨动量
        # 击穿3日均线代表微观结构上的恐慌情绪开始瓦解
        vix_exhaustion = vix < vix.rolling(window=3).mean()
        
        # 3. 跨资产确认条件: 黄金(终极避风港)的期权波动率也必须同步衰竭，确认系统性恐慌退潮
        gvz_exhaustion = gvz < gvz.rolling(window=3).mean()

        # 核心触发逻辑: 极度狂飙 + 本身动量衰竭 + 跨资产交叉确认衰竭
        long_condition = (vix_zscore > 2.5) & vix_exhaustion & gvz_exhaustion

        # 严格执行脉冲赋值，常态维持 0.0
        signal.loc[long_condition] = 1.0

        signal.name = self.name
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}()"