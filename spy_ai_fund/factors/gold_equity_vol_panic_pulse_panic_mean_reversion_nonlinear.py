import numpy as np
import pandas as pd

class GoldEquityVolPanicPulseFactor:
    """跨资产波动率极值与衰竭因子 (panic_mean_reversion/nonlinear)

    逻辑: 黄金波动率(GVZ)反映避险资产挤兑，股票波动率(VIX)反映风险资产恐慌。当两者同时达到历史极值时，说明市场处于极端的"现金为王"流动性冲击(Dash for Cash)。此时一旦双波动的短期动能由正转负，标志恐慌抛售正式见顶衰竭，触发强胜率抄底买入(+1.0)；反之，若两者在常态水平上方稳步共振上升且未达极值，说明跨资产避险情绪在缓慢发酵，风险传染扩大，输出看空(-1.0)。
    数据: vixcls, gvzcls
    输出: +1.0 表示系统性恐慌见顶衰竭(安全买点)，-1.0 表示风险跨资产恶化初期(钝刀割肉期)
    触发条件: 双Z-Score极值交叉 + 一阶动量反转，预期 Trigger Rate: 5%-15%
    """

    def __init__(self, z_window=126, extreme_z=1.5, mild_z=0.5):
        self.name = 'gold_equity_vol_panic_pulse'
        # 126个交易日(约半年)能迅速反映结构性的短期恐慌脉冲基准
        self.z_window = z_window
        # 统计学阈值，1.5代表单侧剧烈偏离(极端爆发)，0.5代表初现上行端倪
        self.extreme_z = extreme_z
        self.mild_z = mild_z

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 前向填充缺失值以保持边际动量计算的连续性
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 动态 Z-Score (126日)
        vix_mean = vix.rolling(window=self.z_window, min_periods=21).mean()
        vix_std = vix.rolling(window=self.z_window, min_periods=21).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-6)
        
        gvz_mean = gvz.rolling(window=self.z_window, min_periods=21).mean()
        gvz_std = gvz.rolling(window=self.z_window, min_periods=21).std()
        gvz_z = (gvz - gvz_mean) / (gvz_std + 1e-6)
        
        # 边际变化：二阶导数防御法则
        vix_diff_1d = vix.diff(1)
        gvz_diff_1d = gvz.diff(1)
        
        vix_diff_3d = vix.diff(3)
        gvz_diff_3d = gvz.diff(3)
        
        # 多头条件：恐慌处于极致区域 + 今日同时出现回落 + 连续3日丧失上行势头 (抄底接飞刀防范)
        bull_cond = (
            (vix_z > self.extreme_z) & 
            (gvz_z > self.extreme_z) & 
            (vix_diff_1d < 0) & 
            (gvz_diff_1d < 0) & 
            (vix_diff_3d <= 0)
        )
        
        # 空头条件：恐慌处于轻微偏离均值 + 正在跨资产共振上升发酵 + 还没达到衰竭极点
        bear_cond = (
            (vix_z > self.mild_z) & (vix_z <= self.extreme_z) &
            (gvz_z > self.mild_z) & (gvz_z <= self.extreme_z) &
            (vix_diff_1d > 0) & 
            (gvz_diff_1d > 0) &
            (vix_diff_3d > 0) &
            (gvz_diff_3d > 0)
        )
        
        signal.loc[bull_cond] = 1.0
        signal.loc[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_window={self.z_window}, extreme_z={self.extreme_z}, mild_z={self.mild_z})"