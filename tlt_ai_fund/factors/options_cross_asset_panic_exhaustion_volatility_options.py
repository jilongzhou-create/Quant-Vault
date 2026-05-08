import numpy as np
import pandas as pd

class VolatilityCurveRegimeReversalFactor:
    """Volatility Curve Regime Reversal (volatility/options)

    逻辑: 将期权波动率的极值反转(VIX恐慌衰竭/VIX平淡惊醒)与国债收益率曲线的边际变化(牛平/牛陡/熊平/熊陡)相结合。
          单纯的波动率高低并不能决定美债的涨跌, 必须结合宏观定价环境(跨域确认):
          1. 恐慌衰竭 (VIX极高并回升转跌):
             - 若前期为"熊平"(加息/通胀恐慌): 紧缩预期落地, 债市超跌反弹 -> 看多 (+1.0)
             - 若前期为"牛陡"(衰退/避险恐慌): 避险情绪消退, 资金流出债市 -> 看空 (-1.0)
             - 若前期为"熊陡"(期限溢价恐慌): 抛售潮竭尽, 债市企稳 -> 看多 (+1.0)
             - 若前期为"牛平"(降息周期末端): 衰竭意味着利好出尽 -> 看空 (-1.0)
          2. 平淡惊醒 (VIX极低并止跌回升):
             - 伴随"熊平"(加息突袭起步): 债市面临紧缩冲击 -> 看空 (-1.0)
             - 伴随"牛陡"(衰退初现端倪): 避险资金涌入债市 -> 看多 (+1.0)
             - 伴随"熊陡"(供给冲击起步): 债市面临抛压 -> 看空 (-1.0)
             - 伴随"牛平"(降息抢跑起步): 债市受追捧 -> 看多 (+1.0)
    数据: vixcls (波动率), t10y2y (收益率曲线), dgs2 (短端利率用于界定牛熊)
    触发: VIX 126日 Z-Score > 1.25 或 < -1.25 且发生一阶导数拐点, 配合 10日曲线边际变动 > 5bps
    输出: +1.0 或 -1.0 的狙击手级脉冲信号
    """

    def __init__(self):
        self.name = 'volatility_curve_regime_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        required_cols = ['vixcls', 't10y2y', 'dgs2']
        if not all(col in data.columns for col in required_cols):
            return signal

        # 1. 数据预处理
        vix = data['vixcls'].ffill()
        curve = data['t10y2y'].ffill()
        short_rate = data['dgs2'].ffill()

        # 2. 波动率极端与衰竭计算 (遵守零值休眠与二阶导数铁律)
        vix_roll_mean = vix.rolling(window=126).mean()
        vix_roll_std = vix.rolling(window=126).std()
        vix_z = (vix - vix_roll_mean) / (vix_roll_std + 1e-6)

        vix_diff = vix.diff()
        vix_ma3 = vix.rolling(window=3).mean()

        # 高波动率衰竭 (恐慌结束)
        high_vol_exhaust = (vix_z > 1.25) & (vix_diff < 0) & (vix < vix_ma3)
        
        # 低波动率惊醒 (平淡结束)
        low_vol_reversal = (vix_z < -1.25) & (vix_diff > 0) & (vix > vix_ma3)

        # 3. 收益率曲线宏观状态确认 (遵守边际变化铁律, 5bps = 0.05%)
        curve_diff = curve.diff(10)
        short_diff = short_rate.diff(10)

        # 定义四大曲线变动状态
        bull_steep = (curve_diff > 0.05) & (short_diff < -0.05)
        bear_flat = (curve_diff < -0.05) & (short_diff > 0.05)
        bear_steep = (curve_diff > 0.05) & (short_diff > 0.05)
        bull_flat = (curve_diff < -0.05) & (short_diff < -0.05)

        # 4. 生成脉冲信号 (仅在拐点瞬间且有宏观确认时触发)
        
        # 恐慌结束场景
        signal[high_vol_exhaust & bull_steep] = -1.0  # 衰退恐慌避险盘撤退
        signal[high_vol_exhaust & bear_flat] = 1.0    # 加息恐慌超跌反弹
        signal[high_vol_exhaust & bear_steep] = 1.0   # 抛售恐慌竭尽反弹
        signal[high_vol_exhaust & bull_flat] = -1.0   # 极致宽松预期退潮

        # 平淡惊醒场景
        signal[low_vol_reversal & bull_steep] = 1.0   # 衰退避险刚启动
        signal[low_vol_reversal & bear_flat] = -1.0   # 紧缩惊吓刚启动
        signal[low_vol_reversal & bear_steep] = -1.0  # 通胀/供给惊吓起步
        signal[low_vol_reversal & bull_flat] = 1.0    # 宽松抢跑起步

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"VolatilityCurveRegimeReversalFactor(name={self.name})"