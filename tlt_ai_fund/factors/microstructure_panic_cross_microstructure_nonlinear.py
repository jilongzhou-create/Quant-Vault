import numpy as np
import pandas as pd

class MicrostructurePanicCrossFactor:
    """恐慌与微观流动性挤兑同步衰竭因子 (microstructure/nonlinear)

    逻辑: 结合股市恐慌(VIX)与避险资产流动性挤兑(GVZ黄金波动率或TLT成交量)。在深度流动性危机中(如2020年3月)，股市与避险资产均遭遇无差别抛售导致波动率齐升；一旦VIX极端冲高后向下击穿3日均线形成脉冲拐点，且交叉验证维度也确认回落，表明抛售高潮见顶、流动性恢复，此时抄底美债能吃到最确定的反弹。严格的脉冲击穿逻辑(二阶导数)确保不在高波主跌浪中接飞刀。
    数据: vixcls, gvzcls (若有volume则结合volume)
    触发: VIX近5日Z-Score > 2.0 且今日下穿3日均线 + 交叉维度极值且处于回落状态
    输出: +1.0 (恐慌衰竭脉冲做多), -1.0 (自满打破脉冲做空), 零值休眠
    """

    def __init__(self):
        self.name = 'microstructure_panic_cross_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (默认全为 0.0)
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        
        # 经济学含义: 252日为一整年，衡量年度级别的宏观情绪水位
        vix_mean = vix.rolling(window=252, min_periods=63).mean()
        vix_std = vix.rolling(window=252, min_periods=63).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-6)
        
        # 铁律2: 二阶导数条件1 - 处于极端水位 (过去5天内曾达到恐慌高位 > 2.0，或自满低位 < -1.5)
        vix_extreme_long = (vix_z > 2.0).rolling(window=5, min_periods=1).max() > 0
        vix_extreme_short = (vix_z < -1.5).rolling(window=5, min_periods=1).max() > 0
        
        # 铁律2/3: 二阶导数条件2 & 边际变化 - 捕捉衰竭瞬间的 Sniper Pulse 脉冲
        vix_ma3 = vix.rolling(window=3, min_periods=1).mean()
        # 今天刚向下击穿3日均线 (今天小于，且昨天大于等于)，绝对禁止常态输出
        vix_exhaustion_pulse_long = (vix < vix_ma3) & (vix.shift(1) >= vix_ma3.shift(1))
        # 今天刚向上打破3日均线
        vix_exhaustion_pulse_short = (vix > vix_ma3) & (vix.shift(1) <= vix_ma3.shift(1))
        
        # 寻找次级维度进行非线性交叉验证 (首选黄金波动率GVZ，备选微观成交量)
        sec = None
        if 'gvzcls' in data.columns:
            sec = data['gvzcls'].ffill()
            window_len = 252
            z_long, z_short = 1.5, -1.0
        elif 'volume' in data.columns:
            sec = data['volume'].ffill()
            window_len = 63  # 成交量用季度窗口更敏感
            z_long, z_short = 1.5, -1.0
            
        if sec is not None:
            sec_mean = sec.rolling(window=window_len, min_periods=21).mean()
            sec_std = sec.rolling(window=window_len, min_periods=21).std()
            sec_z = (sec - sec_mean) / (sec_std + 1e-6)
            
            # 次级维度极值验证
            sec_extreme_long = (sec_z > z_long).rolling(window=5, min_periods=1).max() > 0
            sec_extreme_short = (sec_z < z_short).rolling(window=5, min_periods=1).max() > 0
            
            # 次级维度衰竭验证
            sec_ma3 = sec.rolling(window=3, min_periods=1).mean()
            sec_is_exhausted_long = (sec < sec_ma3)
            sec_is_worsening_short = (sec > sec_ma3)
            
            long_cond = vix_extreme_long & sec_extreme_long & vix_exhaustion_pulse_long & sec_is_exhausted_long
            short_cond = vix_extreme_short & sec_extreme_short & vix_exhaustion_pulse_short & sec_is_worsening_short
        else:
            # 缺失次级维度时，退化为单一高阈值脉冲
            long_cond = vix_extreme_long & vix_exhaustion_pulse_long
            short_cond = vix_extreme_short & vix_exhaustion_pulse_short
            
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"