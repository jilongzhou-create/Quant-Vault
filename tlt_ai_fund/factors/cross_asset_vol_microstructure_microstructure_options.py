import numpy as np
import pandas as pd

class CrossAssetVolMicrostructureFactor:
    """跨资产期权波动率微观结构因子 (microstructure/options)

    逻辑: VIX(美股隐含波动率)与GVZ(黄金隐含波动率)的利差衡量了跨资产的流动性压力不对称性。当该利差极端飙升时，表明市场处于“抛售一切换取现金”的纯流动性恐慌状态。一旦利差极端后见顶回落（衰竭），标志着无差别抛售结束，避险资金将重新涌入美债，触发多头脉冲。
    数据: vixcls, gvzcls
    触发: VIX-GVZ 利差的 252日 Z-Score > 2.5 且当天利差小于过去3日均值
    输出: +1.0 表示流动性恐慌衰竭，脉冲抄底美债；常态严格为 0.0
    """

    def __init__(self, zscore_window: int = 252, zscore_threshold: float = 2.5, decay_window: int = 3):
        self.name = 'cross_asset_vol_microstructure'
        self.zscore_window = zscore_window
        self.zscore_threshold = zscore_threshold
        self.decay_window = decay_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始全 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 处理数据缺失的情况
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 前向填充缺失值以防止计算中断
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产期权隐含波动率利差
        vol_spread = vix - gvz
        
        # 计算 252 日滚动 Z-Score (代表一年的交易日基准)
        min_periods = self.zscore_window // 2
        roll_mean = vol_spread.rolling(window=self.zscore_window, min_periods=min_periods).mean()
        roll_std = vol_spread.rolling(window=self.zscore_window, min_periods=min_periods).std()
        
        # 避免除以 0 的异常
        roll_std = roll_std.replace(0, np.nan)
        z_score = (vol_spread - roll_mean) / roll_std
        
        # 铁律2: 二阶导数 (衰竭条件), 必须等指标开始回落才触发信号
        decay_condition = vol_spread < vol_spread.rolling(window=self.decay_window).mean()
        
        # 条件1: 处于极端高位
        extreme_panic = z_score > self.zscore_threshold
        
        # 两个条件同时满足触发多头脉冲 (脉冲结束后自动回归 0.0)
        buy_pulse = extreme_panic & decay_condition
        
        # 赋值信号
        signal[buy_pulse] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, zscore_threshold={self.zscore_threshold}, decay_window={self.decay_window})"