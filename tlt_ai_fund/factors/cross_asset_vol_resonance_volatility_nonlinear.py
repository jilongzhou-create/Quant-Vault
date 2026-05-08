import numpy as np
import pandas as pd

class CrossAssetVolResonanceFactor:
    """跨资产波动率极值共振衰竭因子 (volatility/nonlinear)

    逻辑: 单一资产(如VIX)飙升时美债可能遭遇流动性冲击被同向抛售，不能直接买入。本因子交叉观测权益(VIX)和终极避险资产黄金(GVZ)的波动率，当任一波动率陷入极度恐慌(一季度Z-Score>1.5)且两者开始同步衰竭回落时，标志着无差别抛售引发的流动性危机真正结束，避险配置盘全面回流美债，产生胜率极高的做多脉冲。反之，当极度低波后双双向上突破时，风险平价策略被迫降杠杆抛售股债，产生看空脉冲。该因子常态为0，为精准的狙击手级反转信号。
    数据: vixcls (VIX指数), gvzcls (黄金波动率指数)
    触发: 极值条件(Z-Score > 1.5) + 二阶衰竭条件(diff() < 0 且 跌破3日均值) 交叉共振 -> +1.0
    输出: 稀疏脉冲信号 [-1.0, 1.0]，正值代表危机解除做多TLT，负值代表低波破位做空TLT
    """

    def __init__(self, lookback_window: int = 63, zscore_threshold: float = 1.5, momentum_window: int = 3):
        self.name = 'cross_asset_vol_resonance_volatility_nonlinear'
        self.lookback = lookback_window
        self.z_threshold = zscore_threshold
        self.mom_window = momentum_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['vixcls', 'gvzcls']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # Forward fill missing values to maintain temporal continuity for daily diffs
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 1. 经济学特征计算: 63日 (单季度) 波动率偏离度 (Z-Score)
        vix_mean = vix.rolling(window=self.lookback).mean()
        vix_std = vix.rolling(window=self.lookback).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std
        
        gvz_mean = gvz.rolling(window=self.lookback).mean()
        gvz_std = gvz.rolling(window=self.lookback).std().replace(0, np.nan)
        gvz_z = (gvz - gvz_mean) / gvz_std
        
        # 2. 衰竭与边际变化判定 (铁律2 & 铁律3: 必须有负向 diff 及均值失守确认)
        vix_exhaustion = (vix.diff() < 0) & (vix < vix.rolling(window=self.mom_window).mean())
        gvz_exhaustion = (gvz.diff() < 0) & (gvz < gvz.rolling(window=self.mom_window).mean())
        
        # 3. 突发飙升判定 (用于打破极致拥挤的低波动状态)
        vix_breakout = (vix.diff() > 0) & (vix > vix.rolling(window=self.mom_window).mean())
        gvz_breakout = (gvz.diff() > 0) & (gvz > gvz.rolling(window=self.mom_window).mean())
        
        # 4. 非线性交叉触发逻辑: 极值 + 跨资产确认 + 衰竭
        # 做多TLT: 任一维度发生极端恐慌，且跨资产波动率同步确认衰竭 (流动性休克结束)
        extreme_panic = (vix_z > self.z_threshold) | (gvz_z > self.z_threshold)
        long_cond = extreme_panic & vix_exhaustion & gvz_exhaustion
        
        # 做空TLT: 任一维度发生极端自满(低波动拥挤)，且跨资产波动率同步爆发 (Risk-Parity 降杠杆前兆)
        extreme_complacency = (vix_z < -self.z_threshold) | (gvz_z < -self.z_threshold)
        short_cond = extreme_complacency & vix_breakout & gvz_breakout
        
        # 5. 零值休眠铁律: 严格脉冲赋值
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback={self.lookback}, z_thresh={self.z_threshold}, mom_win={self.mom_window})"