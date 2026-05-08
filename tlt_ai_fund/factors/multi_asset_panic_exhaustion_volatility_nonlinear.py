import numpy as np
import pandas as pd

class MultiAssetPanicExhaustionFactor:
    """多资产恐慌衰竭因子 (volatility/nonlinear)

    逻辑: 监控股票恐慌(VIX)与避险资产恐慌(GVZ黄金波动率)的同步极端狂飙。在流动性挤兑期，股金同步遭抛售导致波动率双双极值；当双波动率同时衰竭回落，且美债收益率曲线出现陡峭化动量（短端下行确认宽松），表明跨资产去杠杆结束，输出强烈的看多美债脉冲。反之，在极度自满且曲线平坦化时看空。
    数据: vixcls (VIX), gvzcls (黄金波动率), t10y2y (期限利差)
    触发: VIX 252日Z-Score>2.5且跌破3日均线 + GVZ Z-Score>2.0且跌破3日均线 + t10y2y 5日动量>0 -> +1.0
    输出: 狙击手级脉冲信号。常态为 0.0，极端恐慌瓦解日为 +1.0，极度自满反转日为 -1.0。
    """

    def __init__(self):
        self.name = 'multi_asset_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 数据校验与预处理
        required_cols = ['vixcls', 'gvzcls', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        df = data[required_cols].ffill()
        signal = pd.Series(0.0, index=df.index, name=self.name)

        vix = df['vixcls']
        gvz = df['gvzcls']
        t10y2y = df['t10y2y']

        # 2. 宏观长周期 Z-Score (252个交易日)
        vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
        gvz_z = (gvz - gvz.rolling(252).mean()) / gvz.rolling(252).std()

        # 3. 多头触发逻辑 (恐慌挤兑极值 + 二阶导数衰竭 + 边际曲线陡峭化)
        # 铁律2: 必须等衰竭! 过去5天内VIX达到>2.5的极值，且"今天"跌破3日均线开始回落
        vix_extreme_high = vix_z.rolling(5).max() > 2.5
        vix_exhaustion_down = vix < vix.rolling(3).mean()

        # 跨资产确认: 黄金波动率(避险端流动性挤兑)也同样达到极值并衰竭
        gvz_extreme_high = gvz_z.rolling(5).max() > 2.0
        gvz_exhaustion_down = gvz < gvz.rolling(3).mean()

        # 铁律3: 边际变化。利差5日差分为正 (Bull Steepening, 降息预期抢跑短端)
        curve_steepening = t10y2y.diff(5) > 0.0

        long_trigger = (vix_extreme_high & vix_exhaustion_down & 
                        gvz_extreme_high & gvz_exhaustion_down & 
                        curve_steepening)

        # 4. 空头触发逻辑 (自满极值 + 二阶导数向上反转 + 边际曲线平坦化)
        # 波动率极度低迷(Z < -1.5)后开始反弹，意味着"温水煮青蛙"结束，风险偏好收缩
        vix_extreme_low = vix_z.rolling(5).min() < -1.5
        vix_exhaustion_up = vix > vix.rolling(3).mean()

        gvz_extreme_low = gvz_z.rolling(5).min() < -1.5
        gvz_exhaustion_up = gvz > gvz.rolling(3).mean()

        # 利差5日差分为负 (Bear Flattening, 紧缩/加息预期升温)
        curve_flattening = t10y2y.diff(5) < 0.0

        short_trigger = (vix_extreme_low & vix_exhaustion_up & 
                         gvz_extreme_low & gvz_exhaustion_up & 
                         curve_flattening)

        # 5. 铁律1: 零值休眠，只在特定日脉冲触发
        signal[long_trigger] = 1.0
        signal[short_trigger] = -1.0

        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"