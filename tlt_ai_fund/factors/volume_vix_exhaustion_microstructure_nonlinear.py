import numpy as np
import pandas as pd

class VolumeVixExhaustionFactor:
    """微观结构成交量与宏观恐慌的非线性衰竭反转因子 (microstructure/nonlinear)

    逻辑: 结合了微观层面的 TLT ETF 极端爆量抛售 (Capitulation) 与宏观层面的 VIX 极致恐慌。
          策略严格遵守二阶导数反转铁律：绝对禁止在恐慌飙升途中接飞刀。只有当微观成交量在近期出现极值（爆量恐慌）、
          且当前宏观 VIX 仍处极高位时，同时两者在边际上双双出现明显衰竭（VIX回落且低于3日均线，微观抛售缩量退潮），
          才确认为流动性冲击见顶，此时输出单日高胜率看多脉冲。
    数据: volume (TLT微观成交量), vixcls (VIX波动率)
    触发: vixcls Z-Score > 2.5 且 .diff() < 0 且低于3日均值；同时近期 volume 曾触发 Z-Score > 2.5 且当前显著缩量。
    输出: +1.0 (仅在极度恐慌衰竭点输出极短期看多脉冲)
    """

    def __init__(self, vix_window: int = 252, vol_window: int = 63, z_threshold: float = 2.5):
        self.name = 'vol_vix_exhaustion_pulse'
        self.vix_window = vix_window       # 252个交易日，对应约一年宏观状态
        self.vol_window = vol_window       # 63个交易日，对应约一个季度微观基准
        self.z_threshold = z_threshold     # 2.5 标准差，捕捉尾部极端冲击

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 常态下信号必须保持为 0.0 的零值休眠状态
        signal = pd.Series(0.0, index=data.index)
        
        # 拦截数据缺失
        if 'vixcls' not in data.columns or 'volume' not in data.columns:
            return signal
            
        # 防止含有 NaN 导致整列计算失真
        vix = data['vixcls'].ffill()
        vol = data['volume'].ffill()
        
        # --- 条件一：计算极值水位 (避免魔法数字，使用滚动 Z-Score) ---
        
        # 计算 VIX 的极端极值 (长周期)
        vix_mean = vix.rolling(window=self.vix_window, min_periods=self.vix_window//2).mean()
        vix_std = vix.rolling(window=self.vix_window, min_periods=self.vix_window//2).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)
        
        # 计算 Volume 的微观极端抛售极值 (中周期)
        vol_mean = vol.rolling(window=self.vol_window, min_periods=self.vol_window//2).mean()
        vol_std = vol.rolling(window=self.vol_window, min_periods=self.vol_window//2).std()
        vol_z = (vol - vol_mean) / (vol_std + 1e-8)
        
        # --- 条件二：二阶导数与边际变化 (防接飞刀，捕捉衰竭) ---
        
        # VIX 当前仍处于极高位置
        vix_extreme = vix_z > self.z_threshold
        # 铁律2: VIX 必须出现衰竭，边际回落且跌破极短均线
        vix_exhaust = (vix.diff() < 0) & (vix < vix.rolling(3).mean())
        
        # 微观抛售极值：过去3天内曾发生过爆量强平（成交量脉冲极短，允许错位回落）
        vol_extreme_recent = (vol_z > self.z_threshold).rolling(window=3, min_periods=1).max() > 0
        # 铁律2/3: 当前微观成交量已经缩量衰竭，做空动能彻底退潮
        vol_exhaust = vol < vol.rolling(5).mean()
        
        # --- 最终交叉过滤 ---
        
        # 严苛触发逻辑：宏观恐慌且见顶回落 AND 微观曾爆量且当前枯竭
        trigger = vix_extreme & vix_exhaust & vol_extreme_recent & vol_exhaust
        
        # 输出脉冲看多信号
        signal[trigger] = 1.0
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(vix_window={self.vix_window}, vol_window={self.vol_window}, z_threshold={self.z_threshold})"