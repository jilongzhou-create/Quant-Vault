import numpy as np
import pandas as pd

class MicrostructureCollateralScarcityReversalFactor:
    """微观结构流动性交叉反转 (Microstructure / Nonlinear)

    逻辑: 危机期间流动性干涸，资金疯狂涌入前端无风险抵押品(3个月T-Bill)，导致 DFF-DTB3 利差剧烈飙升(微观资金面极度恐慌)。由于DFF是阶梯状政策利率，必须使用利差的边际变化(动量)来捕捉突发冲击。当利差动量Z-Score极高且VIX极高，随后两者同步回落(衰竭)，表明央行已注入流动性(如扩表/降息)，微观挤兑结束，此时是长端美债(TLT)的极佳抄底买点。
    数据: dff, dtb3, vixcls
    触发: DFF-DTB3利差的3日变化率 Z-Score > 2.5 且开始回落，交叉 VIX Z-Score > 2.0 且开始回落.
    输出: +1.0 看多美债, -1.0 看空美债 (严格遵循零值休眠的脉冲型信号)
    """

    def __init__(self):
        self.name = 'microstructure_collateral_reversal'
        self.window = 252

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 严格遵守铁律1: 常态下必须为 0.0，零值休眠
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['dff', 'dtb3', 'vixcls']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 基础数据前向填充，防止空值
        dff = data['dff'].ffill()
        dtb3 = data['dtb3'].ffill()
        vix = data['vixcls'].ffill()
        
        # 铁律3: 边际变化。对于DFF等阶梯状政策利率，禁止使用绝对水位，必须使用边际动量
        # 衡量前端流动性避险溢价的变化率 (3日动量捕捉脉冲冲击)
        tbs_spread = dff - dtb3
        spread_change = tbs_spread.diff(3)
        
        # 计算流动性微观压力的 Z-Score
        spread_mean = spread_change.rolling(window=self.window, min_periods=20).mean()
        spread_std = spread_change.rolling(window=self.window, min_periods=20).std()
        spread_std = spread_std.replace(0.0, np.nan)
        spread_zscore = (spread_change - spread_mean) / spread_std
        
        # 计算宏观恐慌 VIX 的 Z-Score
        vix_mean = vix.rolling(window=self.window, min_periods=20).mean()
        vix_std = vix.rolling(window=self.window, min_periods=20).std()
        vix_std = vix_std.replace(0.0, np.nan)
        vix_zscore = (vix - vix_mean) / vix_std
        
        # 铁律1: 极值条件 (狙击手级触发门槛)
        micro_extreme_long = spread_zscore > 2.5
        micro_extreme_short = spread_zscore < -2.5
        
        vol_extreme_long = vix_zscore > 2.0
        vol_extreme_short = vix_zscore < -2.0
        
        # 铁律2: 二阶导数衰竭 (绝对禁止接飞刀，极值必须叠加回落确认)
        micro_exhaust_long = spread_change < spread_change.rolling(3).mean()
        vol_exhaust_long = vix < vix.rolling(3).mean()
        
        micro_exhaust_short = spread_change > spread_change.rolling(3).mean()
        vol_exhaust_short = vix > vix.rolling(3).mean()
        
        # 方法C: 非线性特征交叉 (多重恐慌指标同步极值且同步衰竭)
        long_cond = micro_extreme_long & micro_exhaust_long & vol_extreme_long & vol_exhaust_long
        short_cond = micro_extreme_short & micro_exhaust_short & vol_extreme_short & vol_exhaust_short
        
        # 生成脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window})"