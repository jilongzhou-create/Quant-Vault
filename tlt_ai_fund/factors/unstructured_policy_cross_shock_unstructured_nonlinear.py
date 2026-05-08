import numpy as np
import pandas as pd

class UnstructuredMacroPivotFactor:
    """Unstructured Macro Pivot & Curve Resonator (Unstructured / Nonlinear)

    逻辑: 挖掘非结构化文本数据(经济政策不确定性 EPU)与金融压力(FSI)的极值反转，并要求美债曲线形态给出严格的右侧共振。
         当市场极度恐慌(不确定性/压力激增 Z>2.0)并开始衰竭回落时，若伴随短端利率反弹与曲线平坦化(Bear Flattening)，说明避险情绪消退、降息预期受挫，触发Risk-On看空美债脉冲。
         反之，当极度自满(指数急降 Z<-2.0)开始向上反转时，若伴随短端利率下行与曲线变陡(Bull Steepening)，说明风险重新觉醒、降息预期急升，触发Risk-Off看多美债脉冲。
    数据: usepuindxd (经济政策不确定性), stlfsi4 (金融压力指数), dgs2 (2年期美债), t10y2y (期限利差).
    触发: EPU或FSI的5日动量极值(Z>2.0) + 衰竭(穿越3日均线) + 收益率与曲线动量共振确认.
    输出: [-1.0, 0.0, 1.0] 严格的狙击手级脉冲信号.
    """

    def __init__(self):
        self.name = 'unstructured_macro_pivot_factor'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # Check required columns to avoid KeyError (CoreAnchor fields dfii10/dgs10/bamlh0a0hym2 are STRICTLY AVOIDED)
        required_cols = ['usepuindxd', 'stlfsi4', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        # Forward fill to handle weekends or slight daily reporting misalignments
        epu = data['usepuindxd'].ffill()
        fsi = data['stlfsi4'].ffill()
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # 边际变化铁律 (Marginal Change Iron Rule): 
        # Calculate 5-day momentum for robust structural shifts instead of daily noise
        epu_chg5 = epu.diff(5)
        fsi_chg5 = fsi.diff(5)
        dgs2_chg5 = dgs2.diff(5)
        t10y2y_chg5 = t10y2y.diff(5)

        # Calculate Z-scores over a 63-day (~1 FICC quarter) rolling window to capture local regime extremes
        window = 63
        epu_z = (epu_chg5 - epu_chg5.rolling(window).mean()) / (epu_chg5.rolling(window).std() + 1e-8)
        fsi_z = (fsi_chg5 - fsi_chg5.rolling(window).mean()) / (fsi_chg5.rolling(window).std() + 1e-8)

        # 二阶导数/衰竭铁律 (Anti-Catch-Falling-Knife Iron Rule):
        # Must cross the 3-day moving average to confirm the momentum has broken
        epu_ma3 = epu.rolling(3).mean()
        fsi_ma3 = fsi.rolling(3).mean()
        
        epu_exh_down = epu < epu_ma3  # Panic/Surge is subsiding
        epu_exh_up = epu > epu_ma3    # Complacency/Drop is ending
        
        fsi_exh_down = fsi < fsi_ma3
        fsi_exh_up = fsi > fsi_ma3

        # Curve Confirmations (Nonlinear feature cross with bond market pricing)
        # Bull Steepening: Short end dropping, curve steepening (Dovish pivot / Safe haven demand)
        bull_steepening = (dgs2_chg5 < 0) & (t10y2y_chg5 > 0)
        # Bear Flattening: Short end rising, curve flattening (Hawkish pricing / Risk normalization)
        bear_flattening = (dgs2_chg5 > 0) & (t10y2y_chg5 < 0)

        # 零值休眠铁律 (Sniper Pulse Iron Rule): Default to 0.0
        signal = pd.Series(0.0, index=data.index)

        # LONG TLT (+1.0): Risk-Off Awakening
        # Condition 1: Extreme complacency shock (Z < -2.0) in either Uncertainty or Stress
        # Condition 2: Complacency Exhaustion (Value ticks up above 3-day MA)
        # Condition 3: Confirmed by Bull Steepening in the Treasury curve
        long_cond_epu = (epu_z < -2.0) & epu_exh_up
        long_cond_fsi = (fsi_z < -2.0) & fsi_exh_up
        long_trigger = (long_cond_epu | long_cond_fsi) & bull_steepening

        # SHORT TLT (-1.0): Risk-On Normalization
        # Condition 1: Extreme panic shock (Z > 2.0) in either Uncertainty or Stress
        # Condition 2: Panic Exhaustion (Value ticks down below 3-day MA)
        # Condition 3: Confirmed by Bear Flattening in the Treasury curve
        short_cond_epu = (epu_z > 2.0) & epu_exh_down
        short_cond_fsi = (fsi_z > 2.0) & fsi_exh_down
        short_trigger = (short_cond_epu | short_cond_fsi) & bear_flattening

        # Assign pulse values
        signal[long_trigger] = 1.0
        signal[short_trigger] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"