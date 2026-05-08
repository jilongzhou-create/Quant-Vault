import numpy as np
import pandas as pd

class CrossAssetVolDislocationReversalFactor:
    """跨资产波动率错位反转因子 (volatility/options)

    逻辑: FICC市场对跨资产波动率的"错位"高度敏感。当VIX(权益恐慌)与GVZ(黄金/实物恐慌)的比值出现极端偏离时, 
          意味着单一市场的强平流动性冲击达到顶峰。根据二阶导数铁律, 我们绝对不接飞刀, 而是等待这种极度错位
          (Z-Score > 2.5 或 < -2.5) 开始衰竭 (比值拐头且绝对波动率下降) 时才触发脉冲。
          无论流动性恐慌还是通胀恐慌的瓦解, 都会促使资金重新理性配置美债, 触发看多脉冲(+1.0)。
          反之, 当整体波动率极度受到压抑(贪婪极致)且突然苏醒时, 风险平价基金的无差别去杠杆会导致股债双杀, 触发看空脉冲(-1.0)。
    数据: vixcls (VIX指数), gvzcls (黄金波动率指数)
    触发: 
      多头(+1.0): 波动率比值(VIX/GVZ)的252日Z-Score > 2.5 (或<-2.5) 且 比值开始均值回归 且 绝对波动率衰竭回落(<3日均值)
      空头(-1.0): VIX的252日Z-Score < -2.0 且 跨资产波动率(VIX和GVZ)同步开始飙升(>3日均值)
    输出: 狙击手级脉冲信号, 常态严格为0.0, 仅在错位瓦解瞬间输出 +1.0/-1.0。
    """

    def __init__(self):
        self.name = 'cross_asset_vol_dislocation_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'gvzcls']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 避免除以0的极少数异常情况
        gvz_safe = gvz.replace(0, np.nan).ffill()
        
        # 计算跨资产波动率比值 (VIX vs Gold Vol)
        vol_ratio = vix / gvz_safe
        
        # 核心指标计算 (252个交易日 Z-Score, 最少需要63天数据)
        ratio_mean = vol_ratio.rolling(window=252, min_periods=63).mean()
        ratio_std = vol_ratio.rolling(window=252, min_periods=63).std()
        ratio_z = (vol_ratio - ratio_mean) / ratio_std
        
        vix_mean = vix.rolling(window=252, min_periods=63).mean()
        vix_std = vix.rolling(window=252, min_periods=63).std()
        vix_z = (vix - vix_mean) / vix_std
        
        # 边际变化参考: 3日均值与单日Diff，用于判断二阶导数衰竭
        vix_ma3 = vix.rolling(window=3).mean()
        gvz_ma3 = gvz.rolling(window=3).mean()
        
        # ---------------------------------------------------------------------
        # 铁律2: 二阶导数衰竭条件 (绝不直接在极值做多，必须等拐点)
        # ---------------------------------------------------------------------
        
        # 情形1: 权益流动性恐慌极度错位 (VIX飙升远超GVZ) -> 开始瓦解
        liq_panic_extreme = ratio_z > 2.5
        liq_panic_exhaustion = (vol_ratio.diff() < 0) & (vix < vix_ma3) & (vix.diff() < 0)
        
        # 情形2: 实物/通胀恐慌极度错位 (GVZ飙升远超VIX) -> 开始瓦解
        inf_panic_extreme = ratio_z < -2.5
        inf_panic_exhaustion = (vol_ratio.diff() > 0) & (gvz < gvz_ma3) & (gvz.diff() < 0)
        
        # 多头信号: 任何一种极度恐慌的瓦解，都意味着美债流动性恢复或加息预期降温，利好TLT
        long_cond = (liq_panic_extreme & liq_panic_exhaustion) | (inf_panic_extreme & inf_panic_exhaustion)
        
        # ---------------------------------------------------------------------
        # 情形3: 极度贪婪/波动率压抑 -> 突然苏醒 (风险平价去杠杆，股债双杀)
        # ---------------------------------------------------------------------
        
        # 波动率极度低迷 (Z < -2.0, 因为波动率存在下限且右偏，-2.0已是非常极端的压抑)
        complacency_extreme = vix_z < -2.0
        # 边际变化: 跨资产波动率同步飙升，打破沉寂状态
        complacency_breakout = (vix > vix_ma3) & (vix.diff() > 0) & (gvz > gvz_ma3) & (gvz.diff() > 0)
        
        short_cond = complacency_extreme & complacency_breakout
        
        # ---------------------------------------------------------------------
        # 铁律1: 零值休眠 (仅在事件发生时脉冲触发)
        # ---------------------------------------------------------------------
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        # 填充NaN为0，确保完全干净的脉冲Series
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"