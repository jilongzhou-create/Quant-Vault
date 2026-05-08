import numpy as np
import pandas as pd

class CreditVixPanicReversionNonlinearFactor:
    """恐慌均值回归 (panic_mean_reversion/nonlinear)

    逻辑: 美股具有典型的"长牛+急跌+均值回归"物理属性。当波动率(VIX)或信用利差(HY OAS)进入历史前10%-15%的高危区间(Z-Score>1.2)时代表极度恐慌；若此时两者的单日与3日边际同时转负，证明高压情绪正式见顶衰竭，触发强烈抄底买入(+1.0)；而当两者处于均值之上但未到极端的区间(0<Z<=1.2)且边际连续上升时，属于温水煮青蛙式的趋势恶化，触发看空(-1.0)。
    数据: vixcls (VIX), bamlh0a0hym2 (高收益债信用利差)
    输出: +1.0 看多 (恐慌见顶衰竭)，-1.0 看空 (趋势持续恶化)，0.0 常态休眠
    触发条件: 极值(Z>1.2)且动量转负时输出+1.0，均值上方(0<Z<=1.2)且动量向上时输出-1.0。预期 Trigger Rate 8-12%。
    """

    def __init__(self, window=252, z_threshold=1.2):
        self.name = 'credit_vix_panic_reversion_nonlinear'
        # 252个交易日代表一年的自然滚动回溯周期
        self.window = window
        # Z=1.2 约对应正态分布的Top 11.5%，具有明确的"极端尾部风险"经济学含义
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 如果缺少核心列，直接返回休眠信号 0.0
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        # 前向填充缺失值以防止节假日导致的数据断层
        vix = data['vixcls'].ffill()
        hy = data['bamlh0a0hym2'].ffill()
        
        # 计算一年期滚动均值和标准差
        vix_mean = vix.rolling(window=self.window, min_periods=60).mean()
        vix_std = vix.rolling(window=self.window, min_periods=60).std()
        hy_mean = hy.rolling(window=self.window, min_periods=60).mean()
        hy_std = hy.rolling(window=self.window, min_periods=60).std()
        
        # 计算 Z-Score，定位风险的极值状态
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        hy_z = (hy - hy_mean) / hy_std.replace(0, np.nan)
        
        # 计算边际变化 (1日动量和3日动量)
        vix_diff1 = vix.diff(1)
        hy_diff1 = hy.diff(1)
        vix_diff3 = vix.diff(3)
        hy_diff3 = hy.diff(3)
        
        # 初始化全零脉冲信号
        signal = pd.Series(0.0, index=data.index)
        
        # +1.0 强看多条件 (极度恐慌 + 极值见顶衰竭)
        # 1. 处于极度恐慌区间 (VIX或HY信用利差的Z-Score大于1.2)
        # 2. 当天开始回落 (diff1 < 0) 且 3天短期动量也转负 (diff3 < 0)，确认衰竭
        buy_cond_extreme = (vix_z > self.z_threshold) | (hy_z > self.z_threshold)
        buy_cond_exhaustion = (vix_diff1 < 0) & (hy_diff1 < 0) & (vix_diff3 < 0) & (hy_diff3 < 0)
        buy_mask = buy_cond_extreme & buy_cond_exhaustion
        
        # -1.0 强看空条件 (轻度恐慌 + 趋势恶化)
        # 1. 恐慌情绪抬头，但未到接飞刀的极值区间 (0 < Z-Score <= 1.2)
        # 2. 边际恐慌正在加剧，未见顶 (1天和3天动量均大于0)
        sell_cond_mild = (vix_z > 0.0) & (vix_z <= self.z_threshold) & (hy_z > 0.0) & (hy_z <= self.z_threshold)
        sell_cond_worsening = (vix_diff1 > 0) & (hy_diff1 > 0) & (vix_diff3 > 0) & (hy_diff3 > 0)
        sell_mask = sell_cond_mild & sell_cond_worsening
        
        # 应用二阶导数交叉逻辑
        signal.loc[buy_mask] = 1.0
        signal.loc[sell_mask] = -1.0
        
        signal.name = self.name
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold})"