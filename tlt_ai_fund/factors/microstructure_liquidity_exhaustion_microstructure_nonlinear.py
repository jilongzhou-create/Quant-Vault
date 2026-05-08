import numpy as np
import pandas as pd

class MicrostructureLiquidityExhaustionFactor:
    """微观流动性与恐慌抛售非线性共振衰竭 (microstructure/nonlinear)

    逻辑: 捕捉底层流动性危机导致的微观抛售衰竭。当宏观金融压力(nfci/stlfsi4)处于极端危排水位，且微观层面TLT成交量(volume)或跨资产恐慌(VIX)呈现极值激增时，市场处于强制平仓的主跌浪。一旦这些高频压力指标同步产生二阶导向下拐点(边际回落或低于短期均值)，表明微观流动性挤兑和抛压同时衰竭，形成胜率极高的非线性反转脉冲，触发抄底美债信号。
    数据: nfci (或 stlfsi4), vixcls, volume
    触发: [金融压力 Z-Score > 1.5 且开始边际回落] AND [VIX或成交量 Z-Score > 2.0 且边际回落]
    输出: +1.0 (极值见顶回落后的看多脉冲)，常态下严格保持 0.0 (Sniper Pulse)
    """

    def __init__(self):
        self.name = 'microstructure_liquidity_exhaustion_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 常态休眠，全 0.0 序列初始化
        signal = pd.Series(0.0, index=data.index)
        
        # 1. 提取核心微观流动性/金融压力指标 (首选 nfci，备选 stlfsi4)
        if 'nfci' in data.columns:
            stress = data['nfci'].ffill()
        elif 'stlfsi4' in data.columns:
            stress = data['stlfsi4'].ffill()
        else:
            # 缺失基石数据直接返回休眠信号
            return signal
            
        # 计算流动性压力 252日(一年) Z-Score
        stress_mean = stress.rolling(window=252, min_periods=63).mean()
        stress_std = stress.rolling(window=252, min_periods=63).std()
        stress_zscore = (stress - stress_mean) / (stress_std + 1e-8)
        
        # 铁律2 & 3: 衰竭与边际变化条件 (二阶导数反转)
        # 金融压力指标为低频/易出现平台期的数据，用 < 5日均值 来确认真实的边际回落拐点
        stress_exhaustion = (stress.diff() < 0) | (stress < stress.rolling(window=5).mean())
        # 设置 1.5 的阈值结合极其苛刻的微观共振条件，以确保 Trigger Rate 落在 5%-15% 的合理区间
        cond_stress = (stress_zscore > 1.5) & stress_exhaustion
        
        # 2. 提取微观抛售与跨资产恐慌特征
        cond_panic = pd.Series(False, index=data.index)
        has_panic_data = False
        
        # 交叉特征 A: 跨资产波动率恐慌衰竭
        if 'vixcls' in data.columns:
            vix = data['vixcls'].ffill()
            vix_mean = vix.rolling(window=252, min_periods=63).mean()
            vix_std = vix.rolling(window=252, min_periods=63).std()
            vix_zscore = (vix - vix_mean) / (vix_std + 1e-8)
            
            # VIX 见顶衰竭: 当日回落 或 跌破3日移动均线 (拒接飞刀)
            vix_exhaustion = (vix.diff() < 0) | (vix < vix.rolling(window=3).mean())
            cond_panic = cond_panic | ((vix_zscore > 2.0) & vix_exhaustion)
            has_panic_data = True
            
        # 交叉特征 B: 标的微观层面 (TLT ETF) 的强制平仓放量衰竭
        if 'volume' in data.columns:
            vol = data['volume'].ffill()
            # 微观放量使用 63日(一季度) 滚动窗口来反映近期极值
            vol_mean = vol.rolling(window=63, min_periods=21).mean()
            vol_std = vol.rolling(window=63, min_periods=21).std()
            vol_zscore = (vol - vol_mean) / (vol_std + 1e-8)
            
            # 成交量衰竭: 巨量抛售后量能开始萎缩
            vol_exhaustion = (vol.diff() < 0) | (vol < vol.rolling(window=3).mean())
            cond_panic = cond_panic | ((vol_zscore > 2.0) & vol_exhaustion)
            has_panic_data = True
            
        if not has_panic_data:
            # 若双指标皆缺失，退阶提高压力指标的阈值以保证严谨性
            cond_panic = (stress_zscore > 2.5)
            
        # 3. 方法C: 高维非线性特征交叉 (金融系统压力见顶 AND 市场微观恐慌同时见顶衰竭)
        trigger = cond_stress & cond_panic
        
        # 4. 触发看多脉冲
        signal[trigger] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"