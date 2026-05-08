import numpy as np
import pandas as pd

class CrossAssetVolSpreadFactor:
    """跨资产隐含波动率利差因子 (microstructure/options)

    逻辑: VIX(权益恐慌)与GVZ(黄金恐慌)的利差代表流动性危机与滞胀/地缘危机的相对强度。
          当利差极端飙升(股市崩盘主导导致Cash is King)且开始回落时，流动性压力缓解，资金从现金配置回流安全久期资产，看多美债(+1.0)；
          当利差极端下挫(滞胀/地缘导致黄金疯抢)且开始反弹时，避险情绪消退，资金追逐风险资产从而抛售美债，看空美债(-1.0)。
          常态下因子保持静默，属于极值反转狙击脉冲。
    数据: vixcls (CBOE VIX 波动率指数), gvzcls (CBOE 黄金 ETF 隐含波动率指数)
    触发: VIX-GVZ利差 252日 Z-Score > 2.5 且当天利差小于3日均值 -> +1.0
          VIX-GVZ利差 252日 Z-Score < -2.5 且当天利差大于3日均值 -> -1.0
    输出: [-1.0, 1.0] 的极短期狙击手脉冲信号
    """

    def __init__(self, zscore_window=252, extreme_threshold=2.5, exhaust_window=3):
        self.name = 'cross_asset_vol_spread'
        self.zscore_window = zscore_window
        self.extreme_threshold = extreme_threshold
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 常态下信号必须严格为 0.0 (Sniper Pulse)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必需的数据列是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 提取数据并前向填充，确保无断点
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产期权波动率利差
        spread = vix - gvz
        
        # 计算利差的 252日 Z-Score (避免前瞻偏差)
        min_periods = self.zscore_window // 2
        spread_mean = spread.rolling(window=self.zscore_window, min_periods=min_periods).mean()
        spread_std = spread.rolling(window=self.zscore_window, min_periods=min_periods).std()
        
        # 避免除零错误
        spread_std = spread_std.replace(0.0, np.nan)
        zscore = (spread - spread_mean) / spread_std
        
        # 铁律2: 二阶导数 (极值条件 + 衰竭条件)
        
        # 条件1: 极值条件 (过去3天内 Z-Score 曾达到过极端水平)
        high_extreme = zscore.rolling(window=self.exhaust_window).max() >= self.extreme_threshold
        low_extreme = zscore.rolling(window=self.exhaust_window).min() <= -self.extreme_threshold
        
        # 铁律3: 边际变化
        # 条件2: 衰竭条件 (当前值脱离极端状态，反向穿越短期均值)
        spread_ma = spread.rolling(window=self.exhaust_window).mean()
        high_exhaustion = spread < spread_ma  # 利差见顶回落
        low_exhaustion = spread > spread_ma   # 利差见底反弹
        
        # 触发脉冲信号
        # 多头脉冲: 流动性恐慌到达极点后开始衰竭，资金重新购入美债
        long_trigger = high_extreme & high_exhaustion
        
        # 空头脉冲: 滞胀/地缘恐慌极点衰竭后，资金回流高风险资产抛售美债
        short_trigger = low_extreme & low_exhaustion
        
        signal[long_trigger] = 1.0
        signal[short_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"CrossAssetVolSpreadFactor(zscore_window={self.zscore_window}, threshold={self.extreme_threshold})"