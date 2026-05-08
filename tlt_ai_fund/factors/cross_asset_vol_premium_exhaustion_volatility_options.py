import numpy as np
import pandas as pd

class CrossAssetVolPremiumExhaustionFactor:
    """跨资产波动率溢价衰竭因子 (volatility/options)

    逻辑: 监控股市期权恐慌(VIX)相对黄金期权避险恐慌(GVZ)的波动率溢价。当该溢价狂飙表明流动性恐慌走向极致，其瓦解瞬间意味着被动去杠杆结束，资金开始向美债(TLT)进行确定性避险配置；相反，当溢价处于极低位反弹时，表明自满情绪破裂，重定价风险将打压债市。脉冲输出完美避免单边市的"接飞刀"。
    数据: vixcls, gvzcls
    触发: 溢价 126日 Z-Score > 2.5 且 衰竭(diff < 0 且 低于3日均值) -> 恐慌衰竭看多 (+1.0)；Z-Score < -2.0 且 飙升(diff > 0 且 高于3日均值) -> 自满破裂看空 (-1.0)。
    输出: 严格脉冲信号 [-1.0, 1.0]，常态为 0.0。
    """

    def __init__(self):
        self.name = 'cross_asset_vol_premium_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理缺失的数据列
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 获取数据并前向填充处理停牌/假期
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()

        # 核心逻辑：计算跨资产期权隐含波动率溢价 (股市恐慌 - 黄金避险恐慌)
        vol_premium = vix - gvz
        
        # 计算 126日 (半年期) 的动态 Z-Score，捕捉中短期内的极端情绪挤压
        window = 126
        vp_mean = vol_premium.rolling(window=window).mean()
        vp_std = vol_premium.rolling(window=window).std().replace(0, np.nan)
        z_score = (vol_premium - vp_mean) / vp_std

        # 边际变化与微观结构衰竭条件 (二阶导数)
        vp_diff = vol_premium.diff()
        vp_ma3 = vol_premium.rolling(window=3).mean()

        # 初始化零值休眠 Series
        signal = pd.Series(0.0, index=data.index)

        # --------------------------------------------------------------------------------
        # 铁律1 & 铁律2 & 铁律3 执行：极值 + 衰竭 + 边际变化
        # --------------------------------------------------------------------------------
        
        # 脉冲做多 (+1.0)：
        # 条件1 (极值)：Z-Score > 2.5 (流动性恐慌处于极致高点)
        # 条件2 (衰竭)：diff < 0 且 当前值回落至 3日均值下方 (抛压瓦解，做多美债)
        long_trigger = (z_score > 2.5) & (vp_diff < 0) & (vol_premium < vp_ma3)
        
        # 脉冲做空 (-1.0)：
        # 条件1 (极值)：Z-Score < -2.0 (市场极度自满，忽视股市相对风险)
        # 条件2 (反转)：diff > 0 且 当前值反弹至 3日均值上方 (恐慌突然复苏，重定价紧缩预期，做空美债)
        short_trigger = (z_score < -2.0) & (vp_diff > 0) & (vol_premium > vp_ma3)

        # 信号赋值
        signal.loc[long_trigger] = 1.0
        signal.loc[short_trigger] = -1.0

        # 清理由于均值/方差引发的早期 NaN
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window=126, z_threshold_long=2.5, z_threshold_short=-2.0)"