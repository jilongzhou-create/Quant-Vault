import numpy as np
import pandas as pd

class RatePivotLiquidityPulseFactor:
    """Rate Pivot Liquidity Pulse (policy_pivot/nonlinear)

    逻辑: 捕捉美债短端利率(DGS2)与期限利差(T10Y2Y)动量变化引发的流动性拐点。
          短端利率是美联储政策预期的核心锚。当短端利率大幅下行且收益率曲线变陡时(Bull Steepening)，意味着市场正在抢跑鸽派降息，这对美股是强烈的流动性利好；
          相反，当短端利率跳升且曲线变平(Bear Flattening)时，代表鹰派加息冲击，压制美股估值。
          对于极端的加息恐慌(Z-Score > 2.5)或极端的衰退恐慌(Z-Score < -2.5)，因子严格遵守“极值+衰竭”二阶导数铁律，在恐慌停止恶化的瞬间反向看多(抄底)。
    数据: dgs2, t10y2y
    输出: +1.0 (流动性宽松或极端恐慌衰竭看多), -1.0 (鹰派冲击看空)
    触发条件: 10日波动Z-Score达到极值且伴随曲线形态确认，同时二阶导数(加速度)发出脉冲触发信号。预期Trigger Rate 8% - 12%。
    """

    def __init__(self):
        self.name = 'rate_pivot_liquidity_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查必要字段是否存在
        required_cols = ['dgs2', 't10y2y']
        if not all(col in data.columns for col in required_cols):
            return pd.Series(0.0, index=data.index)

        # 数据前向填充，处理节假日缺失值
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # 计算 10日 边际变化量 (捕捉低频数据的动量跃升)
        dgs2_10d = dgs2.diff(10)
        t10y2y_10d = t10y2y.diff(10)

        # 计算 DGS2 动量的动态 Z-Score (使用252个交易日窗口，至少一季度数据)
        dgs2_10d_mean = dgs2_10d.rolling(window=252, min_periods=63).mean()
        dgs2_10d_std = dgs2_10d.rolling(window=252, min_periods=63).std()
        
        # 避免除以 0 导致 inf
        dgs2_z = (dgs2_10d - dgs2_10d_mean) / dgs2_10d_std.replace(0, np.nan)

        # 计算二阶导数 (动量的加速度)，用于识别趋势爆发瞬间或恐慌衰竭的拐点
        dgs2_accel = dgs2_10d.diff()

        signal = pd.Series(0.0, index=data.index)

        # =====================================================================
        # 状态 1: 温和鸽派转向 (Benign Dovish Pivot)
        # 短端利率显著下行 (-2.5 < Z <= -1.0)，收益率曲线陡峭化 (t10y2y_10d > 0)，且下行趋势正在加速 (accel < 0)
        # 结论: 市场开始交易软着陆与降息，看多
        buy_mild = (dgs2_z <= -1.0) & (dgs2_z > -2.5) & (t10y2y_10d > 0.0) & (dgs2_accel < 0)

        # 状态 2: 温和鹰派冲击 (Mild Hawkish Shock)
        # 短端利率显著上行 (1.0 <= Z < 2.5)，收益率曲线平坦化 (t10y2y_10d < 0)，且上行趋势正在加速 (accel > 0)
        # 结论: 市场受通胀或鹰派言论惊吓，流动性收紧，看空
        sell_mild = (dgs2_z >= 1.0) & (dgs2_z < 2.5) & (t10y2y_10d < 0.0) & (dgs2_accel > 0)

        # 状态 3: 极端鹰派恐慌衰竭 (Extreme Hawkish Exhaustion - 抄底)
        # 短端利率出现历史罕见的狂飙 (Z >= 2.5)，但今日加速度转负 (accel < 0)，说明单边抛售国债的最恐慌时刻过去
        # 结论: 鹰派利空出尽，均值回归看多
        buy_extreme_hawkish = (dgs2_z >= 2.5) & (dgs2_accel < 0)

        # 状态 4: 极端衰退恐慌衰竭 (Extreme Recession Exhaustion - 抄底)
        # 短端利率出现崩盘式暴跌 (Z <= -2.5)，意味着市场正在定价严重的经济衰退或流动性危机，此时不能接飞刀
        # 必须等待下跌动量开始放缓 (accel > 0)，说明恐慌盘出尽
        # 结论: 危机干预预期上升，衰退恐慌衰竭，抄底看多
        buy_extreme_dovish = (dgs2_z <= -2.5) & (dgs2_accel > 0)
        # =====================================================================

        # 信号合成
        signal.loc[buy_mild | buy_extreme_hawkish | buy_extreme_dovish] = 1.0
        signal.loc[sell_mild] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"