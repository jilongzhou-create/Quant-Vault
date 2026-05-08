import numpy as np
import pandas as pd

class PolicyVolExhaustionFactor:
    """政策不确定性与跨资产波动率极值衰竭因子 (volatility/unstructured)

    逻辑: 经济政策不确定性(USEPUINDXD, 基于非结构化新闻文本计算)极度飙升代表宏观系统性恐慌。单纯恐慌期间买入容易接飞刀，当不确定性与跨资产波动率(VIX/GVZ)从极端水位同步见顶回落时，标志着宏观避险抛售潮瓦解，是配置美债捕捉反转的极佳脉冲时点。
    数据: usepuindxd (经济政策不确定性指数), vixcls (美股波动率), gvzcls (黄金波动率)
    触发: 不确定性 Z-Score > 1.5 且 不确定性/VIX/GVZCLS 同步跌破3日均线 (二阶导数确认)
    输出: 脉冲信号 +1.0(看多美债), -1.0(看空美债), 常态输出 0.0
    """

    def __init__(self, z_window=252, z_thresh=1.5, smooth_win=3):
        self.name = 'policy_vol_exhaustion'
        self.z_window = z_window
        self.z_thresh = z_thresh
        self.smooth_win = smooth_win

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖数据是否存在
        req_cols = ['usepuindxd', 'vixcls', 'gvzcls']
        if not all(col in data.columns for col in req_cols):
            return signal
            
        df = data[req_cols].ffill()
        epu = df['usepuindxd']
        vix = df['vixcls']
        gvz = df['gvzcls']
        
        # 计算 EPU (基于非结构化新闻数据的波动率表征) 的 Z-Score (边际变化铁律)
        epu_mean = epu.rolling(window=self.z_window, min_periods=self.z_window//2).mean()
        epu_std = epu.rolling(window=self.z_window, min_periods=self.z_window//2).std()
        epu_z = (epu - epu_mean) / epu_std
        
        # 衰竭条件 (二阶导数铁律: 极值不能直接买入，必须等衰竭确认)
        epu_falling = epu < epu.rolling(self.smooth_win).mean()
        vix_falling = vix < vix.rolling(self.smooth_win).mean()
        gvz_falling = gvz < gvz.rolling(self.smooth_win).mean()
        
        epu_rising = epu > epu.rolling(self.smooth_win).mean()
        vix_rising = vix > vix.rolling(self.smooth_win).mean()
        gvz_rising = gvz > gvz.rolling(self.smooth_win).mean()
        
        # 触发脉冲信号 (零值休眠铁律: 极端状态+跨资产衰竭确认才出手)
        
        # 多头脉冲: 政策恐慌极度高昂，且跨资产恐慌情绪开始全面瓦解
        long_cond = (epu_z > self.z_thresh) & epu_falling & vix_falling & gvz_falling
        
        # 空头脉冲: 政策不确定性极度低迷(极其自满)，且平静被打破跨资产波动率全面抬头
        short_cond = (epu_z < -self.z_thresh) & epu_rising & vix_rising & gvz_rising
        
        # 赋值狙击手脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_window={self.z_window}, z_thresh={self.z_thresh}, smooth_win={self.smooth_win})"