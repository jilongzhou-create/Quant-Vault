import numpy as np
import pandas as pd

class VixGvzFlightToSafetyReversalFactor:
    """波动率极值与跨资产避险情绪反转 (Volatility Crowding Reversal)

    逻辑: 
    1. 恐慌衰竭 (Panic Exhaustion) -> Risk-On -> 做空美债 (-1.0): 
       当宏观波动率处于极端高位(Z>1.8)，且跨资产波动率(VIX与黄金GVZ)开始同步回落时，表明全市场恐慌情绪见顶消退。此时避险盘极度拥挤且开始瓦解，资金从避险资产回流至风险资产，导致美债收益率快速上行，TLT价格下跌。前次回测失败正是因错误假设避险资金会在此刻买入。
    2. 自满破裂 (Complacency Exhaustion) -> Risk-Off -> 做多美债 (+1.0): 
       当宏观波动率处于极度休眠低位(Z<-1.2，即历史后10%分位)，且跨资产波动率突然同步飙升时，表明风平浪静的市场突遭冲击。对冲盘迅速建立，资金疯狂涌入美债等安全资产，压低收益率，导致TLT价格飙升。
    
    数据: vixcls, gvzcls
    触发: 
    - 脉冲做空 (-1.0): 联合Z-Score > 1.8 且 波动率同步回落 (diff < 0 且低于3日均线)
    - 脉冲做多 (+1.0): 联合Z-Score < -1.2 且 波动率同步飙升 (diff > 0 且高于3日均线)
    """

    def __init__(self):
        self.name = 'vix_gvz_flight_to_safety_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全零脉冲信号，严格遵守"常态信号=0.0"的铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 检查核心依赖是否存在
        if 'vixcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        
        # 1. 计算 VIX 252日滚动 Z-Score，反映波动率长期宏观水位
        vix_mean = vix.rolling(252).mean()
        vix_std = vix.rolling(252).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std
        
        # 计算 3日均线，严格用于二阶衰竭判定
        vix_ma3 = vix.rolling(3).mean()
        
        # 2. 检查并融合跨资产GVZ确认 (黄金波动率同样表征避险恐慌)
        has_gvz = 'gvzcls' in data.columns
        if has_gvz:
            gvz = data['gvzcls'].ffill()
            gvz_mean = gvz.rolling(252).mean()
            gvz_std = gvz.rolling(252).std().replace(0, np.nan)
            gvz_z = (gvz - gvz_mean) / gvz_std
            
            # 使用等权均值合成联合宏观波动率水位，若GVZ历史缺失则平滑降级使用VIX
            macro_vol_z = (vix_z + gvz_z.fillna(vix_z)) / 2.0
            
            gvz_ma3 = gvz.rolling(3).mean()
            # 兼容GVZ数据在早期存在空值的情况
            gvz_fading = ((gvz < gvz_ma3) & (gvz.diff() < 0)) | gvz.isna()
            gvz_breaking = ((gvz > gvz_ma3) & (gvz.diff() > 0)) | gvz.isna()
        else:
            macro_vol_z = vix_z
            gvz_fading = True
            gvz_breaking = True
        
        # --- 核心触发条件生成 ---
        
        # 场景 A: 恐慌衰竭 (Z > 1.8 且 开始回落) -> Risk-On -> 资金流出美债 -> 做空TLT (-1.0)
        panic_extreme = macro_vol_z > 1.8
        vix_fading = (vix < vix_ma3) & (vix.diff() < 0)
        short_condition = panic_extreme & vix_fading & gvz_fading
        
        # 场景 B: 自满破裂 (Z < -1.2 且 突然飙升) -> Risk-Off -> 资金避险买入 -> 做多TLT (+1.0)
        complacency_extreme = macro_vol_z < -1.2
        vix_breaking = (vix > vix_ma3) & (vix.diff() > 0)
        long_condition = complacency_extreme & vix_breaking & gvz_breaking
        
        # --- 赋值脉冲信号 ---
        signal[short_condition] = -1.0
        signal[long_condition] = 1.0
        
        # 安全垫：确保无数据期间和未触发日严格为 0.0
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"