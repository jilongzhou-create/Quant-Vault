import numpy as np
import pandas as pd

class FiccMacroVolExhaustionFactor:
    """FICC Macro Volatility Exhaustion & Hawkish Shock (volatility/nonlinear)

    逻辑: 
    由于单纯基于VIX和期权的因子容易失效且重合度高，本因子构建跨越权益、信用、利率三维度的"合成宏观波动率指数"。
    多头(恐慌消退脉冲): 当三维度合成波动率极高(Z>1.0)，且在同一天权益恐慌(VIX)、信用利差、合成波动率同步开始回落(二阶导<0)，准确狙击流动性挤兑衰竭后的美债报复性反弹。
    空头(鹰派冲击脉冲): 当纯利率波动率极高且继续膨胀(Z>1.0, diff>0)，但权益市场尚未产生联动恐慌(VIX_Z<0.5)，同时伴随收益率曲线熊平(利差<0)与信用利差初显走阔，这捕捉了纯粹的超预期鹰派紧缩瞬间，做空美债。
    数据: vixcls(VIX), bamlc0a4cbbb(BBB信用利差), t10y2y(期限利差).
    触发: 极值条件 + 二阶导(动量/衰竭)同步触发，严格遵守零值休眠铁律。
    """

    def __init__(self):
        self.name = 'ficc_macro_vol_exhaustion_volatility_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 严格遵守铁律1: 常态下必须休眠为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 数据集检查
        req_cols = ['vixcls', 'bamlc0a4cbbb', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # 填充缺失值，确保衍生计算的连续性
        vix = data['vixcls'].ffill()
        credit = data['bamlc0a4cbbb'].ffill()
        rate_spread = data['t10y2y'].ffill()
        
        # 计算派生波动率 (21日平均绝对变动，捕捉隐性趋势波动)
        cred_vol = credit.diff().abs().rolling(21).mean()
        rate_vol = rate_spread.diff().abs().rolling(21).mean()
        
        # 计算 252日 Z-Score，反映极端水位 (加 1e-8 防除零警告)
        vix_z = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-8)
        cred_vol_z = (cred_vol - cred_vol.rolling(252).mean()) / (cred_vol.rolling(252).std() + 1e-8)
        rate_vol_z = (rate_vol - rate_vol.rolling(252).mean()) / (rate_vol.rolling(252).std() + 1e-8)
        
        # 构建跨资产合成宏观恐慌指数及其自身 Z-Score
        panic_idx = vix_z + cred_vol_z + rate_vol_z
        z_panic = (panic_idx - panic_idx.rolling(252).mean()) / (panic_idx.rolling(252).std() + 1e-8)
        
        # --- 多头逻辑: 系统性流动性恐慌触极值后同步衰竭 (遵守铁律2: 防接飞刀) ---
        long_extreme = z_panic > 1.0
        # 必须权益与信用双回落，且总体恐慌回落
        long_exhaust = (vix.diff() < 0) & (credit.diff() < 0) & (panic_idx.diff() < 0)
        long_mask = long_extreme & long_exhaust
        
        # --- 空头逻辑: 纯利率端的鹰派紧缩冲击 (避开一般性VIX恐慌导致的避险买盘) ---
        # 利率波动极速恶化，曲线熊平，权益依然麻木，信用初显走阔
        short_rate_extreme = rate_vol_z > 1.0
        short_rate_expanding = rate_vol.diff() > 0
        short_equity_calm = vix_z < 0.5
        short_bear_flattening = rate_spread.diff() < 0
        short_credit_widen = credit.diff() > 0
        short_mask = short_rate_extreme & short_rate_expanding & short_equity_calm & short_bear_flattening & short_credit_widen
        
        # 输出脉冲信号
        signal.loc[long_mask] = 1.0
        signal.loc[short_mask] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"