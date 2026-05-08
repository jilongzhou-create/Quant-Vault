import numpy as np
import pandas as pd

class MicrostructureUnstructuredGoldVolPulseFactor:
    """微观结构/非结构化黄金波动率恐慌衰竭因子 (Microstructure / Unstructured)

    逻辑: 捕捉 FICC 领域核心避险资产的极端流动性挤兑与衰竭反转。黄金波动率指数(GVZ)飙升意味着极端的“现金为王”恐慌(连黄金等顶级避险资产都在被无差别抛售，微观流动性骤干)。
         根据反飞刀铁律，高波动率主跌浪期间绝对禁止买入，必须等恐慌枯竭。只有当 GVZ 创出极值(Z-Score > 2.5) 且开始边际衰竭回落(低于过去3日均值)时，
         才确认微观无差别抛售潮耗尽，流动性危机解除。此时避险资金迅速回补，触发美债(TLT)的极短期强劲看多脉冲。
    数据: gvzcls (黄金波动率指数，带*高价值核心字段)
    触发: gvzcls 的 252日 Z-Score > 2.5 且当日值 < 过去3日均值 (恐慌衰竭)
    输出: +1.0 (抛压极值衰竭，看多美债)，保持极短几天以满足 5%-15% Trigger Rate
    """

    def __init__(self, zscore_window=252, threshold=2.5, exhaust_window=3, hold_days=2):
        self.name = 'microstructure_unstructured_gold_vol_pulse'
        self.zscore_window = zscore_window
        self.threshold = threshold
        self.exhaust_window = exhaust_window
        self.hold_days = hold_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理数据缺失情况，防止报错
        if 'gvzcls' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        # 提取黄金波动率数据并向前填充，避免空值
        gvz = data['gvzcls'].ffill()
        
        # 铁律2: 二阶导数 (极端水位判定)
        gvz_mean = gvz.rolling(window=self.zscore_window, min_periods=20).mean()
        gvz_std = gvz.rolling(window=self.zscore_window, min_periods=20).std()
        
        # 避免分母为 0 产生 inf
        gvz_std = gvz_std.replace(0.0, np.nan)
        z_score = (gvz - gvz_mean) / gvz_std
        
        extreme_panic = z_score > self.threshold
        
        # 铁律2 & 3: 二阶导数与边际变化判定 (衰竭条件，必须等指标回落才能抄底，防接飞刀)
        gvz_rolling_3 = gvz.rolling(window=self.exhaust_window, min_periods=1).mean()
        panic_exhaustion = gvz < gvz_rolling_3
        
        # 产生脉冲触发信号: 必须同时满足极值高位与衰竭回落
        cond_buy = extreme_panic & panic_exhaustion
        
        # 铁律1: 零值休眠 (Sniper Pulse)，非触发日信号严格为 0.0
        raw_signal = pd.Series(0.0, index=data.index)
        raw_signal.loc[cond_buy] = 1.0
        
        # 控制目标 Trigger Rate 在 5% 到 15% 之间
        # 极短期脉冲：通过向前填充，将触发日的脉冲维持极短几天，随后恢复休眠
        signal = raw_signal.replace(0.0, np.nan).ffill(limit=self.hold_days).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, threshold={self.threshold}, exhaust_window={self.exhaust_window})"