import numpy as np
import pandas as pd

class CrossAssetVolReversalNonlinearFactor:
    """跨资产波动率极端拥挤与反转 (volatility/nonlinear)

    逻辑: 监控 VIX 与黄金波动率 (GVZ) 的跨资产联动。恐慌飙升至极值且同步回落时，标志着流动性危机瓦解与避险资金开始实质性回补美债，输出脉冲看多；而在极度自满的平静期后波动率突然抬头，标志着拥挤多头瓦解与抛售压力，输出脉冲看空。脉冲设计确保严格的非连续休眠状态。
    数据: vixcls (VIX指数), gvzcls (黄金波动率指数)
    触发: VIX 63日 Z-Score > 2.5 且3日内发生边际衰竭回落 + GVZ 同步极端且回落 -> +1.0 脉冲；极度平静 (Z < -2.0) 且反弹 -> -1.0 脉冲
    输出: +1.0 表示多头美债脉冲, -1.0 表示空头美债脉冲, 其余非触发日常态为 0.0
    """

    def __init__(self):
        self.name = 'cross_asset_vol_reversal_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠, 初始信号全为0.0
        signal = pd.Series(0.0, index=data.index)

        # 检查所需数据是否在输入 DataFrame 中
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal

        # 填充缺失值，保持最新的真实观测水位
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()

        # 计算 63 个交易日 (约一个季度) 的移动 Z-Score
        vix_mean = vix.rolling(window=63).mean()
        vix_std = vix.rolling(window=63).std()
        vix_z = (vix - vix_mean) / vix_std

        gvz_mean = gvz.rolling(window=63).mean()
        gvz_std = gvz.rolling(window=63).std()
        gvz_z = (gvz - gvz_mean) / gvz_std

        # ==========================================
        # 多头触发逻辑 (铁律2: 二阶导数反转)
        # ==========================================
        # 1. 极端高位条件: 过去3天内曾达到过狂热的恐慌极值
        vix_extreme_recent = vix_z.rolling(window=3).max() > 2.5
        gvz_extreme_recent = gvz_z.rolling(window=3).max() > 2.0
        
        # 2. 衰竭与边际变化条件 (铁律3): 开始明显回落
        # 当天必须是下跌的，且当前值已跌破最近3日移动平均
        vix_exhaustion = (vix.diff() < 0) & (vix < vix.rolling(window=3).mean())
        gvz_exhaustion = (gvz.diff() < 0) & (gvz < gvz.rolling(window=3).mean())

        # 交叉验证: 跨资产恐慌同步衰竭
        long_trigger = vix_extreme_recent & vix_exhaustion & gvz_extreme_recent & gvz_exhaustion

        # ==========================================
        # 空头触发逻辑 (铁律2: 二阶导数反转)
        # ==========================================
        # 1. 极端平静条件: 过去3天内曾处于极度自满的低波动率状态
        vix_calm_recent = vix_z.rolling(window=3).min() < -2.0
        gvz_calm_recent = gvz_z.rolling(window=3).min() < -1.5
        
        # 2. 动量变化条件 (铁律3): 波动率突然抬头飙升
        # 当天必须是上涨的，且当前值突破最近3日均线压制
        vix_spike = (vix.diff() > 0) & (vix > vix.rolling(window=3).mean())
        gvz_spike = (gvz.diff() > 0) & (gvz > gvz.rolling(window=3).mean())

        # 交叉验证: 跨资产波动率同步从底部急剧反弹
        short_trigger = vix_calm_recent & vix_spike & gvz_calm_recent & gvz_spike

        # ==========================================
        # 信号合成
        # ==========================================
        # 脉冲输出，避免主跌浪接飞刀
        signal.loc[long_trigger] = 1.0
        signal.loc[short_trigger] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"