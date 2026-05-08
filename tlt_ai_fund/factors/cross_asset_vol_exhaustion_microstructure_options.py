import numpy as np
import pandas as pd

class CrossAssetVolExhaustionFactor:
    """微观结构/期权波动率跨资产极值衰竭因子 (microstructure/options)

    逻辑: 股市波动率(VIX)与黄金避险波动率(GVZ)的差值代表纯粹的风险资产恐慌溢价。当该溢价达到极端高位并开始回落时, 说明跨资产流动性危机或极端恐慌见顶消退, 市场进入修复期, 避险资金回流稳健生息资产, 触发美债(TLT)的做多脉冲。反之, 当差值处于极端低位(极度自满或通胀恐慌)且开始反转上升时, 预示流动性冲击或加息预期重燃, 触发做空脉冲。必须等待二阶导数拐点以避免接飞刀。
    数据: vixcls, gvzcls
    触发: Z-Score > 2.0 且 差值 < 3日均值 (恐慌衰竭做多); Z-Score < -2.0 且 差值 > 3日均值 (自满反转做空)
    输出: +1.0 (看多美债) / -1.0 (看空美债), 严格脉冲型信号
    """

    def __init__(self, window=126, z_threshold=2.0, smooth_window=3):
        self.name = 'cross_asset_vol_exhaustion'
        self.window = window
        self.z_threshold = z_threshold
        self.smooth_window = smooth_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖列是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 前向填充处理对齐问题 (容忍极短期的缺失)
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产恐慌溢价 (Equity Volatility - Safe Haven Volatility)
        vol_spread = vix - gvz
        
        # 计算局部 Z-Score (捕捉边际极端变化)
        roll_mean = vol_spread.rolling(window=self.window).mean()
        roll_std = vol_spread.rolling(window=self.window).std()
        
        # 避免除以零
        roll_std = roll_std.replace(0.0, np.nan)
        z_score = (vol_spread - roll_mean) / roll_std
        
        # 二阶导数条件: 跨资产恐慌溢价开始回落 (当日值 < 过去3日均值)
        vol_spread_ma3 = vol_spread.rolling(window=self.smooth_window).mean()
        exhaustion_down = vol_spread < vol_spread_ma3
        
        # 二阶导数条件: 极度自满/通胀恐慌开始反转上升 (当日值 > 过去3日均值)
        reversal_up = vol_spread > vol_spread_ma3
        
        # 触发条件1: 极度恐慌且开始衰竭 -> 看多美债脉冲
        long_cond = (z_score > self.z_threshold) & exhaustion_down
        
        # 触发条件2: 极度自满且开始反转 -> 看空美债脉冲
        short_cond = (z_score < -self.z_threshold) & reversal_up
        
        # 赋值狙击手级脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"CrossAssetVolExhaustionFactor(window={self.window}, z_threshold={self.z_threshold}, smooth_window={self.smooth_window})"