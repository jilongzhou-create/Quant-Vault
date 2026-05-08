import numpy as np
import pandas as pd

class VixExhaustionReversalFactor:
    """期权隐含波动率极值衰竭反转 (microstructure/options)

    逻辑: 股市期权隐含波动率(VIX)的极端飙升往往伴随着市场全面的流动性紧缩，所有资产(包括美债)被无差别抛售换现。只有当VIX极值见顶并开始回落时，标志着流动性冲击的结束，避险资金将迅速回流长端美债(TLT)。反之，当极低波动率开始抬升时，平静被打破，长债面临阶段性抛压。因而是典型的极值衰竭脉冲信号。
    数据: vixcls
    触发: 63日 Z-Score > 2.0 且 VIX < 3日均值 (做多)；63日 Z-Score < -1.5 且 VIX > 3日均值 (做空)
    输出: +1.0 (恐慌见顶衰竭，做多美债) / -1.0 (过度自满结束，做空美债)
    """

    def __init__(self):
        self.name = 'vix_exhaustion_reversal'
        self.z_window = 63
        self.smooth_window = 3

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns:
            return signal
            
        # 提取数据并向前填充处理缺失值
        vix = data['vixcls'].ffill()
        
        # 计算微观滚动统计特征 (63日即一个季度，捕捉季度级别的极端脉冲)
        vix_mean = vix.rolling(window=self.z_window).mean()
        vix_std = vix.rolling(window=self.z_window).std()
        
        # 避免除以0
        vix_std = vix_std.replace(0, np.nan)
        vix_zscore = (vix - vix_mean) / vix_std
        
        # 边际变化代理: 当日值与短端平滑均值的关系
        vix_smooth = vix.rolling(window=self.smooth_window).mean()
        
        # 做多脉冲: 极度恐慌 (Z > 2.0) 且 开始衰竭回落 (VIX < 3日均值)
        # 代表无脑抛售阶段结束，真正的避险买盘介入 TLT
        buy_cond = (vix_zscore > 2.0) & (vix < vix_smooth)
        
        # 做空脉冲: 极度自满 (Z < -1.5) 且 平静被打破开始回升 (VIX > 3日均值)
        # 代表风险偏好极高或通胀预期抬头，对无风险长债(TLT)形成抽血效应
        sell_cond = (vix_zscore < -1.5) & (vix > vix_smooth)
        
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"