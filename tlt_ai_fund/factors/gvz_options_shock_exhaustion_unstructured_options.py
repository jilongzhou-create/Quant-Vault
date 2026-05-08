import numpy as np
import pandas as pd

class PivotVolatilityUncertaintyFactor:
    """政策不确定性与期权波动率共振脉冲因子 (unstructured/options)

    逻辑: 结合非结构化新闻不确定性(EPU)与期权隐含波动率(VIX)定位宏观恐慌发酵期。单凭VIX极值无法区分“衰退恐慌(利多美债)”与“通胀恐慌(利空美债)”。因此引入对政策最敏感的2年期美债(DGS2)动量作为方向锚定：若不确定性极值出现，且VIX开始回落、DGS2急降，确认美联储鸽派转向（降息预期陡升），脉冲做多美债(TLT)；若VIX继续攀升且DGS2急升，确认超预期鹰派或通胀失控，脉冲做空美债。
    数据: vixcls, usepuindxd, dgs2
    触发: (VIX或EPU 63日 Z-Score > 0.8) 且 (VIX 3日动量与DGS2 5日动量共振)。非触发日严格输出 0.0。
    输出: 严格在 [-1.0, 1.0] 的脉冲信号。
    """

    def __init__(self):
        self.name = 'pivot_vol_uncertainty_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 信号 (严格遵守零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖数据是否存在
        required_cols = ['vixcls', 'usepuindxd', 'dgs2']
        if not all(col in data.columns for col in required_cols):
            return signal

        # 数据预处理，前向填充以处理非交易日或缺失值
        vix = data['vixcls'].ffill()
        epu = data['usepuindxd'].ffill()
        dgs2 = data['dgs2'].ffill()

        # 1. 宏观不确定性极值评估 (结合 Options 与 Unstructured 数据)
        # 使用 63个交易日 (约一季度) 滚动窗口计算 Z-Score
        # Z-Score > 0.8 约涵盖前 20% 的极端高压区间，保证最终 Trigger Rate 落于 5%-15%
        vix_mean = vix.rolling(window=63, min_periods=21).mean()
        vix_std = vix.rolling(window=63, min_periods=21).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std

        epu_mean = epu.rolling(window=63, min_periods=21).mean()
        epu_std = epu.rolling(window=63, min_periods=21).std().replace(0, np.nan)
        epu_z = (epu - epu_mean) / epu_std

        # 当其中任意一个指标 Z-Score > 0.8 时，确认宏观进入高压或突变状态
        macro_shock = (vix_z > 0.8) | (epu_z > 0.8)

        # 2. 边际变化与二阶导数 (严格遵守极值+衰竭铁律与动量铁律)
        # VIX 动量：判断期权市场恐慌是在消退还是在加剧
        vix_diff = vix.diff(3)
        vix_falling = vix_diff < 0
        vix_rising = vix_diff > 0

        # DGS2 动量：判断 FICC 市场对美联储预期的真实定价方向
        # 选取 5 日变化量过滤日频噪音，阈值设为 5 个基点 (0.05%) 以捕捉实质性的政策突变
        dgs2_diff5 = dgs2.diff(5)
        dovish_pivot = dgs2_diff5 < -0.05
        hawkish_shock = dgs2_diff5 > 0.05

        # 3. 脉冲信号生成
        # 看多美债：宏观高压 + 恐慌开始衰竭(VIX回落) + 降息预期突增(短端急降)
        long_cond = macro_shock & vix_falling & dovish_pivot
        
        # 看空美债：宏观高压 + 恐慌仍在加剧(VIX上升) + 加息预期突增(短端急升)
        short_cond = macro_shock & vix_rising & hawkish_shock

        # 赋值信号并处理逻辑互斥
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        signal[long_cond & short_cond] = 0.0

        # 处理可能产生的 NaN，保障常态输出为 0.0
        signal = signal.fillna(0.0)
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"