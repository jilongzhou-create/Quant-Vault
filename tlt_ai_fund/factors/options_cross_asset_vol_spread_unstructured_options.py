import numpy as np
import pandas as pd

class OptionsCrossAssetVolSpreadFactor:
    """Options Cross-Asset Volatility Spread (unstructured/options)

    逻辑: 捕捉股市恐慌(VIX)与黄金避险恐慌(GVZCLS)的波动率微观结构背离。VIX代表股市流动性恐慌，GVZCLS代表黄金/通胀避险恐慌。
          当 VIX-GVZCLS 差值极端冲高并开始回落时，意味着跨资产流动性恐慌见顶消退，避险资金重新配置美债，输出做多信号(+1.0)；
          当该差值极端探底并开始回升时，意味着纯粹的滞胀/大宗恐慌消退，风险偏好回归，输出做空信号(-1.0)。
          本因子严格遵守零值休眠与二阶导数铁律，仅在极值衰竭瞬间触发脉冲。
    数据: vixcls, gvzcls
    触发: 63日滚动 Z-Score > 2.0 且 差值 < 3日均值 → +1.0；Z-Score < -2.0 且 差值 > 3日均值 → -1.0
    输出: [-1.0, 1.0] 的狙击手级脉冲信号
    """

    def __init__(self):
        self.name = 'options_cross_asset_vol_spread'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号严格为 0.0 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必要字段是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal

        # 填充缺失值
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产波动率差值 (边缘变化 proxy)
        spread = vix - gvz
        
        # 计算滚动 Z-Score (63个交易日约为1个季度，捕捉中期相对冲击)
        window = 63
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std().replace(0, np.nan)
        zscore = (spread - spread_mean) / spread_std
        
        # 二阶导数条件：短期均值作为衰竭判断基准 (Anti-Catch-Falling-Knife)
        spread_ma3 = spread.rolling(window=3).mean()
        
        # 条件1: 差值处于极端高位且开始回落 (流动性恐慌极值已过)
        long_condition = (zscore > 2.0) & (spread < spread_ma3)
        
        # 条件2: 差值处于极端低位且开始回升 (通胀/避险恐慌极值已过)
        short_condition = (zscore < -2.0) & (spread > spread_ma3)
        
        # 触发脉冲信号
        signal.loc[long_condition] = 1.0
        signal.loc[short_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"