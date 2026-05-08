import numpy as np
import pandas as pd

class GlobalFxLiquidityPivotFactor:
    """全球汇率与流动性转向脉冲 (policy_pivot/nonlinear)

    逻辑: 引入汇率维度以重构宏观政策转向的判定，隔离纯利率因子的共线性。单纯的短端利率(2年期)剧烈下行有可能是“衰退恐慌”造成的资金避险(此时伴随美元避险上涨)；只有当2年期美债收益率(降息预期)与广义美元指数(dtwexbgs)同步发生剧烈下跌时，才标志着真正的全球流动性边际宽松和“Risk-on”风险偏好回升，从而触发看多脉冲。反之，两者同步暴涨则为鹰派紧缩与美元流动性收缩的共振冲击，触发看空脉冲。
    数据: [dgs2, dtwexbgs]
    输出: [-1.0, 1.0] 真正的Risk-on宽松共振为+1.0(看多美股), 紧缩冲击共振为-1.0(看空美股)
    触发条件: 2Y收益率与美元指数的5日变化Z-Score同向突破±1.0极值阈值，且当日保持同向动量，预期 Trigger Rate 处于 5-15%。
    """

    def __init__(self):
        self.name = 'global_fx_liquidity_pivot'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 依赖数据字段检查
        req_cols = ['dgs2', 'dtwexbgs']
        for col in req_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        df = data[req_cols].ffill()

        # 计算核心经济变量的5日边际变化
        # 利率使用绝对变动(bps), 汇率使用百分比变动(%)
        dgs2_diff5 = df['dgs2'].diff(5)
        usd_ret5 = df['dtwexbgs'].pct_change(5)

        # 计算1日动量用于确认脉冲延续性(防接飞刀)
        dgs2_diff1 = df['dgs2'].diff(1)
        usd_diff1 = df['dtwexbgs'].diff(1)

        # 动态自适应滚动窗口计算Z-Score (252个交易日)
        dgs2_mean = dgs2_diff5.rolling(window=252, min_periods=60).mean()
        dgs2_std = dgs2_diff5.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        dgs2_z = (dgs2_diff5 - dgs2_mean) / dgs2_std

        usd_mean = usd_ret5.rolling(window=252, min_periods=60).mean()
        usd_std = usd_ret5.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        usd_z = (usd_ret5 - usd_mean) / usd_std

        signal = pd.Series(0.0, index=df.index)

        # 多头信号: 利率预期骤降 (Z < -1.0) 且 美元骤贬 (Z < -1.0) -> 真正的Risk-On宽松
        # 增加具有经济意义的绝对阈值过滤器(-0.10 bps与-0.5%)防止在低波动极寒期产生噪音
        bull_cond = (
            (dgs2_z < -1.0) & 
            (usd_z < -1.0) & 
            (dgs2_diff5 < -0.10) & 
            (usd_ret5 < -0.005) & 
            (dgs2_diff1 < 0) & 
            (usd_diff1 < 0)
        )

        # 空头信号: 利率预期飙升 (Z > 1.0) 且 美元急升 (Z > 1.0) -> 鹰派流动性紧缩与美元虹吸冲击
        bear_cond = (
            (dgs2_z > 1.0) & 
            (usd_z > 1.0) & 
            (dgs2_diff5 > 0.10) & 
            (usd_ret5 > 0.005) & 
            (dgs2_diff1 > 0) & 
            (usd_diff1 > 0)
        )

        signal.loc[bull_cond] = 1.0
        signal.loc[bear_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"