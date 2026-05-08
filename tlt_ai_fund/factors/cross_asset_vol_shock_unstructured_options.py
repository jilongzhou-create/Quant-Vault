import numpy as np
import pandas as pd

class CrossAssetVolShockFactor:
    """跨资产波动率微观结构因子 (unstructured/options)

    逻辑: 纯 VIX 因子在 2022 年失效，因为"通胀恐慌"和"增长恐慌"对美债的影响截然相反。
          本因子通过构建跨资产波动率差值 (VIX 股票恐慌 - GVZ 黄金/通胀恐慌) 来区分宏观环境：
          1. 增长恐慌 (VIX 飙升快于 GVZ): 避险资金涌入美债。当差值回落（恐慌衰竭）时，资金流出避险资产，做空美债 (-1.0)。
          2. 通胀/利率恐慌 (GVZ 飙升快于 VIX): 资金抛售美债。当差值反弹（抛售衰竭）时，利率见顶回落，做多美债 (+1.0)。
    数据: vixcls, gvzcls
    触发: Z-Score > 1.25 (涵盖约10%极值尾部) + 自身波动率绝对值升温 + 跌破/突破 3日均线确认衰竭。
    输出: 脉冲型，常态 0.0，极端衰竭反转日输出 +1.0 或 -1.0。
    """

    def __init__(self):
        self.name = 'unstructured_cross_asset_vol_shock_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 必须存在所需波动率字段
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产波动率差 (股市恐慌 vs 通胀/避险恐慌)
        vol_spread = vix - gvz
        
        # 126日 (半年) 滚动窗口以捕捉中短期宏观体制切换
        window = 126
        
        spread_mean = vol_spread.rolling(window).mean()
        spread_std = vol_spread.rolling(window).std().replace(0, 1e-8)
        z_spread = (vol_spread - spread_mean) / spread_std
        
        vix_mean = vix.rolling(window).mean()
        vix_std = vix.rolling(window).std().replace(0, 1e-8)
        vix_z = (vix - vix_mean) / vix_std
        
        gvz_mean = gvz.rolling(window).mean()
        gvz_std = gvz.rolling(window).std().replace(0, 1e-8)
        gvz_z = (gvz - gvz_mean) / gvz_std
        
        # 二阶导数衰竭条件: 与3日均线交叉，确认趋势已反转
        spread_exhaustion_down = vol_spread < vol_spread.rolling(3).mean()
        spread_exhaustion_up = vol_spread > vol_spread.rolling(3).mean()
        
        # 条件1: 增长恐慌衰竭 (Short TLT)
        # 逻辑: 股市恐慌占据主导 (z_spread > 1.25) 且 VIX 绝对值升温 (vix_z > 0.5)
        # 当 spread 开始回落 (spread_exhaustion_down)，说明避险情绪消退，资金从美债流回股市
        short_cond = (z_spread > 1.25) & (vix_z > 0.5) & spread_exhaustion_down
        
        # 条件2: 通胀/利率恐慌衰竭 (Long TLT)
        # 逻辑: 黄金/通胀恐慌占据主导 (z_spread < -1.25) 且 GVZ 绝对值升温 (gvz_z > 0.5)
        # 当 spread 开始反弹 (spread_exhaustion_up)，说明通胀恐慌见顶，美债利率见顶，资金抄底美债
        long_cond = (z_spread < -1.25) & (gvz_z > 0.5) & spread_exhaustion_up
        
        signal[short_cond] = -1.0
        signal[long_cond] = 1.0
        
        # 清理由于 rolling 造成的 NaN
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"