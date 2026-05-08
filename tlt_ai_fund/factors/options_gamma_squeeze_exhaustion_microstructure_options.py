import numpy as np
import pandas as pd

class OptionsGammaSqueezeExhaustionFactor:
    """期权 Gamma 挤压极值与衰竭反转因子 (microstructure/options)

    逻辑: 捕捉期权市场极端恐慌或对冲盘引发的 Gamma 挤压现象。当 VIX 短期内暴涨（加速度极值）说明做市商强行平仓带来微观流动性冲击；当冲击消退（VIX回落）时，避险资产的连带抛售结束，美债将迎来强劲脉冲式反弹。必须使用脉冲避免接飞刀。
    数据: vixcls (CBOE VIX 波动率指数)
    触发: VIX 5日边际变化量的 252日 Z-Score > 2.5 (极度挤压极值) AND 当日 VIX < 过去3日均值 (恐慌挤压衰竭)
    输出: +1.0 (脉冲看多美债), 其余常态时间严格为 0.0
    """

    def __init__(self, zscore_window: int = 252, diff_window: int = 5, exhaustion_window: int = 3, threshold: float = 2.5):
        self.name = 'options_gamma_squeeze_exhaustion'
        self.zscore_window = zscore_window
        self.diff_window = diff_window
        self.exhaustion_window = exhaustion_window
        self.threshold = threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化常态信号为 0.0 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 处理数据缺失的情况
        if 'vixcls' not in data.columns:
            return signal
            
        # 获取期权隐含波动率数据并处理基础缺失值
        vix = data['vixcls'].ffill()
        
        # 铁律3: 边际变化 - 计算微观对冲挤压指标，即 VIX 短期剧烈变化量，而非绝对水位
        vix_squeeze = vix.diff(self.diff_window)
        
        # 计算边际变化的滚动 Z-Score 极值
        squeeze_mean = vix_squeeze.rolling(self.zscore_window).mean()
        squeeze_std = vix_squeeze.rolling(self.zscore_window).std()
        
        # 防止除以 0 或 NaN
        squeeze_std = squeeze_std.replace(0, np.nan)
        squeeze_zscore = (vix_squeeze - squeeze_mean) / squeeze_std
        
        # 铁律2: 二阶导数 - 极值条件
        extreme_condition = squeeze_zscore > self.threshold
        
        # 铁律2: 二阶导数 - 衰竭条件 (VIX 必须确认跌破近期均值，不再创新高)
        exhaustion_condition = vix < vix.rolling(self.exhaustion_window).mean()
        
        # 狙击手级脉冲：同时满足极值与衰竭才扣动扳机
        buy_trigger = extreme_condition & exhaustion_condition
        
        # 赋值看多信号
        signal[buy_trigger] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"OptionsGammaSqueezeExhaustionFactor(zscore_window={self.zscore_window}, diff_window={self.diff_window}, threshold={self.threshold})"