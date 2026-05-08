import numpy as np
import pandas as pd

class DovishPivotSoftLandingPulseFactor:
    """鸽派转向与软着陆非线性交叉因子 (policy_pivot/nonlinear)

    逻辑: 捕捉真正的宽松拐点而非衰退恐慌。当短端利率急降(预期鸽派)、通胀预期保持坚挺(未进入通缩衰退)、且金融压力指数边际回落时，输出强烈看多脉冲。反之，短端急升、通胀预期未跟升且金融压力边际恶化时，输出鹰派杀估值的看空脉冲。
    数据: dgs2 (2年期国债), t5yie (5年期盈亏平衡通胀), stlfsi4 (金融压力指数)
    输出: +1.0 看多(鸽派软着陆预期), -1.0 看空(鹰派收紧致险), 0.0 观望
    触发条件: DGS2 5日动量Z-Score极值，并结合通胀动量与金融压力动量过滤，预期 Trigger Rate 5%-15%
    """

    def __init__(self):
        self.name = 'dovish_pivot_soft_landing_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        required_cols = ['dgs2', 't5yie', 'stlfsi4']
        
        # 处理数据缺失情况
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        df = data[required_cols].ffill()
        signal = pd.Series(0.0, index=df.index)
        
        # 1. 短端利率 5日变动及其滚动 Z-Score (代表市场短期的流动性预期剧变)
        dgs2_diff = df['dgs2'].diff(5)
        dgs2_diff_std = dgs2_diff.rolling(window=252, min_periods=60).std()
        # 避免除以零
        dgs2_diff_std = dgs2_diff_std.replace(0, np.nan)
        dgs2_z = dgs2_diff / dgs2_diff_std
        
        # 2. 通胀预期 5日变动 (用于区分“正常宽松”与“通缩型衰退恐慌”)
        t5yie_diff = df['t5yie'].diff(5)
        
        # 3. 金融压力指数 5日变动 (用于防接飞刀，危机爆发主跌浪时压力指数会暴涨)
        fsi_diff = df['stlfsi4'].diff(5)
        
        # 【看多触发】: 强流动性宽松预期 + 非衰退恐慌
        # dgs2急跌(Z < -1.2) + 通胀预期未崩盘(> -0.05) + 金融压力正在缓和(< 0.0)
        bull_condition = (
            (dgs2_z < -1.2) & 
            (t5yie_diff > -0.05) & 
            (fsi_diff < 0.0)
        )
        
        # 【看空触发】: 强货币紧缩预期 + 对金融系统造成真实伤害
        # dgs2急升(Z > 1.2) + 并非由通胀暴涨引起(纯鹰派杀估值) + 金融压力正在恶化(> 0.0)
        bear_condition = (
            (dgs2_z > 1.2) & 
            (t5yie_diff < 0.05) & 
            (fsi_diff > 0.0)
        )
        
        signal.loc[bull_condition] = 1.0
        signal.loc[bear_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"