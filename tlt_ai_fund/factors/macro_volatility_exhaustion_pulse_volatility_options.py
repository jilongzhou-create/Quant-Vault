import numpy as np
import pandas as pd

class MacroVolatilityExhaustionPulseFactor:
    """宏观跨资产波动率极值衰竭脉冲因子 (Volatility / Options)

    逻辑: 波动率具有极强的均值回归特性。在恐慌极值期(波动率狂飙), 流动性往往面临无差别抛售(接飞刀)。只有等待期权恐慌(VIX)、收益率曲线波动(t10y2y vol)或政策不确定性(EPU)达到1年期极值(Z>2.5), 且明确出现动量衰竭(二阶导数<0且跌破3日均线)时, 才确认避险资金开始实质性流入美债, 此时输出看多脉冲。反之, 当波动率处于极度低位且开始异动时, 预示宏观风险重估, 输出看空脉冲。因子严格遵循零值休眠与二阶导数铁律。
    数据: vixcls (期权隐含波动率), t10y2y (期限利差, 用于计算曲线动量波动), usepuindxd (经济政策不确定性指数)
    触发:
      - 看多(+1.0): 任意维度的 252日 Z-Score > 2.5, 且当日 diff() < 0, 且低于 3日均值。
      - 看空(-1.0): 任意维度的 252日 Z-Score < -2.0, 且当日 diff() > 0, 且高于 3日均值。
    输出: 严格的狙击手级脉冲信号, 范围 [-1.0, 1.0], 常态为 0.0。
    """

    def __init__(self):
        self.name = 'macro_volatility_exhaustion_pulse'
        self.z_window = 252
        self.cvol_window = 21
        self.ma_window = 3

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 校验底层数据是否存在
        required_cols = ['vixcls', 't10y2y', 'usepuindxd']
        for col in required_cols:
            if col not in data.columns:
                return signal

        # 缺失值前向填充以应对节假日错位
        df = data[required_cols].ffill()

        # 1. 维度一: 美股期权市场恐慌 (Equity Volatility)
        vix = df['vixcls']
        vix_mean = vix.rolling(self.z_window).mean()
        vix_std = vix.rolling(self.z_window).std()
        vix_z = (vix - vix_mean) / vix_std
        vix_ma = vix.rolling(self.ma_window).mean()

        # 2. 维度二: 债市收益率曲线宏观波动率 (Yield Curve Volatility)
        # 使用 10年-2年利差的 21日滚动标准差作为债市动荡的边际变化代理变量
        yc = df['t10y2y']
        cvol = yc.diff().rolling(self.cvol_window).std()
        cvol_mean = cvol.rolling(self.z_window).mean()
        cvol_std = cvol.rolling(self.z_window).std()
        cvol_z = (cvol - cvol_mean) / cvol_std
        cvol_ma = cvol.rolling(self.ma_window).mean()

        # 3. 维度三: 宏观经济政策不确定性 (Policy Uncertainty)
        epu = df['usepuindxd']
        epu_mean = epu.rolling(self.z_window).mean()
        epu_std = epu.rolling(self.z_window).std()
        epu_z = (epu - epu_mean) / epu_std
        epu_ma = epu.rolling(self.ma_window).mean()

        # --- 绝对铁律: 二阶导数衰竭条件计算 (Anti-Catch-Falling-Knife) ---
        
        # 极度恐慌且开始衰竭 -> 看多美债 (避险资金涌入)
        # 严格条件: 1年期Z-Score > 2.5 (极值) AND 较前一日回落 (一阶导<0) AND 跌破短端趋势 (动量破位)
        vix_long = (vix_z > 2.5) & (vix.diff(1) < 0) & (vix < vix_ma)
        cvol_long = (cvol_z > 2.5) & (cvol.diff(1) < 0) & (cvol < cvol_ma)
        epu_long = (epu_z > 2.5) & (epu.diff(1) < 0) & (epu < epu_ma)
        
        long_cond = vix_long | cvol_long | epu_long

        # 极度自满且开始苏醒 -> 看空美债 (风险周期重启, 利率往往上行)
        # 严格条件: 1年期Z-Score < -2.0 (极度平静) AND 较前一日跳升 (一阶导>0) AND 突破短端趋势 (异动确认)
        vix_short = (vix_z < -2.0) & (vix.diff(1) > 0) & (vix > vix_ma)
        cvol_short = (cvol_z < -2.0) & (cvol.diff(1) > 0) & (cvol > cvol_ma)
        epu_short = (epu_z < -2.0) & (epu.diff(1) > 0) & (epu > epu_ma)

        short_cond = vix_short | cvol_short | epu_short

        # 生成零值休眠脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        # 清除多空同时触发的极低概率数据噪音点
        signal[long_cond & short_cond] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_window={self.z_window}, cvol_window={self.cvol_window}, ma_window={self.ma_window})"