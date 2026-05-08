import numpy as np
import pandas as pd

class OptionsVolContagionReversalFactor:
    """Options Volatility Contagion Reversal Factor (volatility/options)

    逻辑: 监控 S&P 500 (VIX) 与黄金 (GVZ) 期权隐含波动率的跨资产恐慌共振。极端的跨资产期权波动率飙升代表市场进入无差别抛售的流动性危机(需追加保证金，此时美债也会被错杀)；当双双从极值回落(二阶导衰竭)时，抛售潮结束，避险资金将疯狂涌入美债，触发脉冲做多。相反，极端自满且波动率开始抬头时，风险平价基金去杠杆机械抛售美债，触发脉冲做空。
    数据: vixcls (CBOE S&P 500 VIX), gvzcls (CBOE Gold VIX)
    触发: 多头=VIX 252日 Z-Score > 2.5 且 GVZ Z-Score > 2.0，且两者期权隐含波动率跌破3日均线(衰竭)；空头=双 Z-Score < -2.0 且向上突破3日均线。
    输出: 狙击手级脉冲信号，+1.0 看多美债，-1.0 看空美债，常态严格为 0.0。
    """

    def __init__(self):
        self.name = 'options_vol_contagion_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 - 初始必须全为 0.0
        signal = pd.Series(0.0, index=data.index)

        # 检查所需数据字段是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal

        # 缺失值前向填充 (避免 Look-ahead bias)
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()

        # 计算宏观基准线: 252个交易日 (约一年) 的滚动统计量
        vix_rolling_mean = vix.rolling(window=252).mean()
        vix_rolling_std = vix.rolling(window=252).std()
        vix_z = (vix - vix_rolling_mean) / vix_rolling_std

        gvz_rolling_mean = gvz.rolling(window=252).mean()
        gvz_rolling_std = gvz.rolling(window=252).std()
        gvz_z = (gvz - gvz_rolling_mean) / gvz_rolling_std

        # 铁律3: 边际变化 - 使用 3 日均线判断短期动能/衰竭方向
        vix_ma3 = vix.rolling(window=3).mean()
        gvz_ma3 = gvz.rolling(window=3).mean()

        # 避免早期数据不足时产生误判
        valid_idx = vix_z.notna() & gvz_z.notna()

        # 铁律2: 二阶导数 - 极值 + 衰竭 (Anti-Catch-Falling-Knife)
        # 多头触发条件: 股市恐慌极度狂飙 (Z>2.5) + 跨资产黄金期权被同步恐慌波及 (Z>2.0) + 两者开始回落 (当前值 < 3日均值)
        long_cond = valid_idx & (vix_z > 2.5) & (gvz_z > 2.0) & (vix < vix_ma3) & (gvz < gvz_ma3)

        # 空头触发条件: 跨资产期权极度自满 (Z<-2.0) + 波动率突然苏醒抬头 (当前值 > 3日均值)
        short_cond = valid_idx & (vix_z < -2.0) & (gvz_z < -2.0) & (vix > vix_ma3) & (gvz > gvz_ma3)

        # 脉冲信号赋值
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"