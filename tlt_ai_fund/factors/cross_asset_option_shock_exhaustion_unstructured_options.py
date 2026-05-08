import numpy as np
import pandas as pd

class CrossAssetOptionShockExhaustionFactor:
    """跨资产期权波动率突变衰竭脉冲因子 (unstructured/options)

    逻辑: 捕捉股票(VIX)和黄金(GVZ)期权隐含波动率的联合极端突变。当跨资产期权波动率同时剧烈飙升时，意味着市场进入极度恐慌和流动性枯竭状态(抛售一切，包括美债)；当这种联合恐慌动量达到极端(Z-Score > 2.5)且开始回落(衰竭)时，标志着央行干预或市场情绪见底，避险资金重新涌入债市，触发强烈看多美债(TLT)的脉冲。反之，若波动率极速暴跌后企稳，意味着风险偏好极度高涨，资金抛弃避险资产，触发看空脉冲。
    数据: vixcls, gvzcls
    触发: 联合波动率3日动量的 252日 Z-Score > 2.5 且开始回落(小于3日均值) -> +1.0； Z-Score < -2.5 且开始企稳(大于3日均值) -> -1.0
    输出: 脉冲型，常态为 0.0
    """

    def __init__(self, lookback_window: int = 252, momentum_days: int = 3, z_threshold: float = 2.5):
        self.name = 'cross_asset_option_shock_exhaustion'
        self.lookback_window = lookback_window
        self.momentum_days = momentum_days
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，默认全 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需数据字段是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 铁律3: 边际变化 (只关注波动率的边际突变，绝对禁止使用波动率绝对值)
        vix_mom = vix.diff(self.momentum_days)
        gvz_mom = gvz.diff(self.momentum_days)
        
        # 计算动量的滚动 Z-Score (反映历史罕见程度)
        vix_z = (vix_mom - vix_mom.rolling(self.lookback_window).mean()) / vix_mom.rolling(self.lookback_window).std()
        gvz_z = (gvz_mom - gvz_mom.rolling(self.lookback_window).mean()) / gvz_mom.rolling(self.lookback_window).std()
        
        # 联合期权恐慌动量 (等权共振)
        iv_shock = (vix_z + gvz_z) / 2.0
        
        # 铁律2: 二阶导数 (衰竭条件，防接飞刀)
        iv_shock_ma3 = iv_shock.rolling(3).mean()
        
        # 多头条件: 恐慌极值 + 衰竭
        # 联合恐慌飙升至极值(>2.5)，且动量开始回落
        long_cond = (iv_shock > self.z_threshold) & (iv_shock < iv_shock_ma3)
        
        # 空头条件: 贪婪极值 + 衰竭
        # 联合波动率极速下降至极值(<-2.5)，且跌势开始企稳反弹
        short_cond = (iv_shock < -self.z_threshold) & (iv_shock > iv_shock_ma3)
        
        # 生成狙击手级脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback={self.lookback_window}, mom_days={self.momentum_days}, z_threshold={self.z_threshold})"