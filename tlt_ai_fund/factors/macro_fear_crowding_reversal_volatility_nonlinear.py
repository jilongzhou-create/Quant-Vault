import numpy as np
import pandas as pd

class MacroFearCrowdingReversalFactor:
    """宏观恐慌与政策不确定性拥挤反转因子 (volatility/nonlinear)

    逻辑: 将金融市场恐慌(VIX)与宏观经济政策不确定性(USEPUINDXD)进行非线性交叉。当双重恐慌指标达到年度极值(Z-Score极端)代表对冲盘极度拥挤，且开始跌破近期均线(二阶导数衰竭)时，系统性危机预期消退，避险情绪及宽松预期兑现，爆发看多美债脉冲。反之在极度平静被打破时产生看空脉冲。
    数据: vixcls, usepuindxd
    触发: VIX(Z>2.5)与USEPUINDXD(Z>2.0)双极端 + 同步跌破3日均线衰竭 -> +1.0 脉冲
    输出: +1.0表示恐慌见顶反转看多美债，-1.0表示宁静被打破看空美债
    """

    def __init__(self, z_vix_long=2.5, z_epu_long=2.0, z_short=-1.5, window=252, smooth=3):
        self.name = 'macro_fear_crowding_reversal_volatility_nonlinear'
        self.z_vix_long = z_vix_long
        self.z_epu_long = z_epu_long
        self.z_short = z_short
        self.window = window
        self.smooth = smooth

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        req_cols = ['vixcls', 'usepuindxd']
        for col in req_cols:
            if col not in data.columns:
                return signal

        # 处理数据缺失，保持前值
        vix = data['vixcls'].ffill()
        epu = data['usepuindxd'].ffill()

        # 计算一年期滚动Z-Score (经济学含义: 中长期波动率水位的相对拥挤度)
        vix_mean = vix.rolling(self.window).mean()
        vix_std = vix.rolling(self.window).std()
        vix_z = (vix - vix_mean) / vix_std

        epu_mean = epu.rolling(self.window).mean()
        epu_std = epu.rolling(self.window).std()
        epu_z = (epu - epu_mean) / epu_std

        # 二阶导数衰竭条件: 极值后的动量反转 (跌破短期均线代表边际恶化停止)
        vix_falling = vix < vix.rolling(self.smooth).mean()
        epu_falling = epu < epu.rolling(self.smooth).mean()

        vix_rising = vix > vix.rolling(self.smooth).mean()
        epu_rising = epu > epu.rolling(self.smooth).mean()

        # 有效性过滤
        valid = vix_std.notna() & (vix_std > 0) & epu_std.notna() & (epu_std > 0)

        # 触发逻辑1: 多维恐慌极端且同步衰竭 -> 看多美债脉冲
        long_cond = valid & (vix_z > self.z_vix_long) & (epu_z > self.z_epu_long) & vix_falling & epu_falling
        
        # 触发逻辑2: 极度平静(做空波动率拥挤)被打破 -> 看空美债脉冲
        short_cond = valid & (vix_z < self.z_short) & (epu_z < self.z_short) & vix_rising & epu_rising

        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_vix={self.z_vix_long}, z_epu={self.z_epu_long}, window={self.window})"