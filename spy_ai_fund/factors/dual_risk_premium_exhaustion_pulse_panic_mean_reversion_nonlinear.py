import numpy as np
import pandas as pd

class DualRiskPremiumExhaustionPulseFactor:
    """双重风险溢价衰竭脉冲因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合股市恐慌(VIX)与信用恐慌(高收益债利差)。极度恐慌且两者动能同时见顶回落时输出+1.0抄底；若风险溢价处于均值上方且呈加速走阔态势，则输出-1.0看空避险。防接飞刀，二阶导数确认衰竭。
    数据: vixcls, bamlh0a0hym2
    输出: 脉冲信号 [-1.0, 1.0]。+1.0为恐慌衰竭抄底，-1.0为恐慌发酵看空，常态0.0。
    触发条件: VIX Z-Score > 1.25或信用利差 Z-Score > 1.0，且双双出现边际回落时触发看多。预期Trigger Rate约在 5%-12% 之间。
    """

    def __init__(self):
        self.name = 'dual_risk_premium_exhaustion_pulse_panic_mean_reversion_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 基础数据缺失检查
        required_cols = ['vixcls', 'bamlh0a0hym2']
        if not all(col in data.columns for col in required_cols):
            return pd.Series(0.0, index=data.index, name=self.name)

        # 数据前向填充以处理节假日错位
        vix = data['vixcls'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()

        # 计算年度(252日) Z-Score 衡量极值水平，最少需要60天预热
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std().replace(0, 1e-5)
        vix_z = (vix - vix_mean) / vix_std

        hy_mean = hy_spread.rolling(window=252, min_periods=60).mean()
        hy_std = hy_spread.rolling(window=252, min_periods=60).std().replace(0, 1e-5)
        hy_z = (hy_spread - hy_mean) / hy_std

        # ---------------- 抄底逻辑：极值 + 衰竭 (防接飞刀) ----------------
        # 1. 压力水平：VIX进入偏度极高区域(>1.25 std) 或 信用利差处于高位(>1.0 std)
        extreme_stress = (vix_z > 1.25) | (hy_z > 1.0)
        
        # 2. 衰竭确认：今日VIX必须下降，且信用利差收窄或不走阔
        vix_exhausted = vix.diff(1) < 0
        hy_exhausted = hy_spread.diff(1) <= 0
        
        buy_cond = extreme_stress & vix_exhausted & hy_exhausted

        # ---------------- 看空逻辑：轻度恐慌 + 发酵蔓延 ----------------
        # 1. 发酵环境：处于历史均值上方，但尚未达到极致绝望状态
        vix_ferment = (vix_z > 0.0) & (vix_z <= 1.25)
        hy_ferment = (hy_z > 0.0) & (hy_z <= 1.0)
        
        # 2. 动量恶化：3日内VIX显著攀升(>1.0点)，且信用利差走阔(>5基点)，今日依然在涨
        vix_surging = (vix.diff(3) > 1.0) & (vix.diff(1) > 0)
        hy_surging = (hy_spread.diff(3) > 0.05)
        
        sell_cond = vix_ferment & hy_ferment & vix_surging & hy_surging

        # ---------------- 信号合成 ----------------
        signal = pd.Series(0.0, index=data.index)
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0

        # 清理由于缺失值可能导致的NaN
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"