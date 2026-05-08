import numpy as np
import pandas as pd

class CrossAssetVolMicrostructureExhaustionFactor:
    """Cross Asset Volatility Microstructure Exhaustion (microstructure/nonlinear)

    逻辑: 当股市波动率(VIX)和黄金波动率(GVZ)同步极端飙升时, 标志着微观结构上跨资产流动性挤兑(Cash is King); 当恐慌极值见顶并同步衰竭回落时, 流动性危机解除, 避险资金重返美债, 触发脉冲做多。反之在极度贪婪反转时做空。
    数据: vixcls, gvzcls
    触发: VIX与GVZ的联合Z-Score > 1.5 (极值), 且联合波动率低于3日均值并伴随当天的负向边际变化 (衰竭反转)
    输出: +1.0 (恐慌见顶衰竭做多), -1.0 (极度贪婪反转做空), 常态输出 0.0
    """

    def __init__(self, window: int = 252, z_threshold: float = 1.5):
        self.name = 'cross_asset_vol_micro_exhaustion_pulse'
        self.window = window
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 必须数据列检查
        required_cols = ['vixcls', 'gvzcls']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 提取并前向填充数据，避免因假期/停盘导致的不一致
        df = data[required_cols].ffill()
        
        vix = df['vixcls']
        gvz = df['gvzcls']
        
        # 计算长周期 Z-Score (使用252个交易日，即一年作为基准锚点)
        vix_mean = vix.rolling(self.window, min_periods=60).mean()
        vix_std = vix.rolling(self.window, min_periods=60).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, 1e-5)
        
        gvz_mean = gvz.rolling(self.window, min_periods=60).mean()
        gvz_std = gvz.rolling(self.window, min_periods=60).std()
        gvz_z = (gvz - gvz_mean) / gvz_std.replace(0, 1e-5)
        
        # 非线性特征交叉: 构建跨资产联合波动率极值评估
        combined_z = (vix_z + gvz_z) / 2.0
        
        # 计算短端动量均值, 满足二阶导数中的 "反转" 铁律
        combined_z_ma3 = combined_z.rolling(3).mean()
        
        # 计算边际变化, 满足边际变化铁律
        vix_diff = vix.diff()
        gvz_diff = gvz.diff()
        
        # 多头触发条件:
        # 1. 跨资产恐慌达到极高水位 (Z-Score > 1.5)
        # 2. 衰竭迹象 (当前值回落到短端3日均线下方)
        # 3. 边际变化为负 (明确的下降脉冲)
        long_cond = (
            (combined_z > self.z_threshold) & 
            (combined_z < combined_z_ma3) & 
            (vix_diff < 0) & 
            (gvz_diff < 0)
        )
        
        # 空头触发条件:
        # 1. 跨资产极度贪婪, 波动率达到极低水位 (Z-Score < -1.5)
        # 2. 反转恶化迹象 (向上突破短端3日均线)
        # 3. 边际变化为正 (波动率开始抬头跳升)
        short_cond = (
            (combined_z < -self.z_threshold) & 
            (combined_z > combined_z_ma3) & 
            (vix_diff > 0) & 
            (gvz_diff > 0)
        )
        
        # 严格的脉冲赋值 (狙击手模式)
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold})"