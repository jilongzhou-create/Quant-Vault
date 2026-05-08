import numpy as np
import pandas as pd

class CrossAssetVolSpreadReversalFactor:
    """跨资产波动率极值反转因子 (volatility/options)

    逻辑: 比较黄金隐含波动率(GVZ)与美股隐含波动率(VIX)的跨资产利差。
          单一的 VIX 因子在 2022 年由于"股债双杀"(通胀恐慌)出现方向性失效。本因子通过跨资产利差区分两种极值状态：
          1. 通缩型恐慌(如2008/2020): VIX 极端飙升远超 GVZ，避险资金涌入导致美债(TLT)暴涨。一旦利差触底回升(恐慌衰竭)，避险资金撤出，此时顺势强烈看空美债(-1.0)。
          2. 通胀/主权信用恐慌(如2022): GVZ 极端飙升压制 VIX，美债遭恐慌性抛售。一旦利差触顶回落(通胀预期见顶衰竭)，美债迎来超跌反弹，此时果断看多美债(+1.0)。
          因子严格遵守二阶导数铁律，仅在极值且开始衰竭的瞬间触发脉冲信号。
    数据: gvzcls (CBOE黄金期权隐含波动率), vixcls (CBOE VIX指数)
    触发: 252日利差 Z-Score > 1.5 且回落低于3日均值 -> 脉冲 +1.0
          252日利差 Z-Score < -1.5 且回升高于3日均值 -> 脉冲 -1.0
    输出: [-1.0, 1.0] 狙击手级别的反转脉冲信号
    """

    def __init__(self):
        self.name = 'cross_asset_vol_spread_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 严格执行零值休眠：初始信号全设为 0.0
        signal = pd.Series(0.0, index=data.index)

        # 检查必要数据列是否存在，缺失则直接返回全0信号
        if 'gvzcls' not in data.columns or 'vixcls' not in data.columns:
            return signal

        # 填充缺失值并提取数据
        gvz = data['gvzcls'].ffill()
        vix = data['vixcls'].ffill()
        
        # 构建核心指标：跨资产波动率利差
        spread = gvz - vix

        # 计算 252 日 (一年期) 滚动宏观 Z-Score
        # 使用 min_periods=63 加速预热
        spread_mean = spread.rolling(window=252, min_periods=63).mean()
        spread_std = spread.rolling(window=252, min_periods=63).std()
        
        # 避免波动率标准差为0导致除以零错误
        spread_std = spread_std.replace(0, np.nan)
        zscore = (spread - spread_mean) / spread_std

        # 边际变化(二阶导数)基准：3日滑动平均线
        spread_ma3 = spread.rolling(window=3, min_periods=1).mean()

        # 核心铁律：极值 + 衰竭确认 (防接飞刀)
        
        # 条件1: 通胀/信用恐慌衰竭 -> 看多美债 (+1.0)
        # 表现为利差极端向上偏离且动量开始衰竭回落
        cond_inflation_exhaust = (zscore > 1.5) & (spread < spread_ma3)
        
        # 条件2: 通缩/股市恐慌衰竭 -> 看空美债 (-1.0)
        # 表现为利差极端向下偏离且动量开始反弹回升
        cond_deflation_exhaust = (zscore < -1.5) & (spread > spread_ma3)

        # 输出脉冲信号
        signal.loc[cond_inflation_exhaust] = 1.0
        signal.loc[cond_deflation_exhaust] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"