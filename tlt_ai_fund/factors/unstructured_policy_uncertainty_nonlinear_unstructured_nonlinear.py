import numpy as np
import pandas as pd

class UnstructuredPolicyUncertaintyNonlinearFactor:
    """政策不确定性与联储文本情绪非线性共振因子 (unstructured/nonlinear)

    逻辑: 结合高频的经济政策不确定性(EPU)极值反转与低频的美联储文本情绪边际变化。当恐慌见顶回落且联储未边际转鸽时，避险消退做空美债；当自满触底反弹且联储未边际转鹰时，避险酝酿做多美债。因子利用极值+二阶反转捕捉情绪周期的瞬时突变，属于典型的狙击手级别脉冲。
    数据: usepuindxd (经济政策不确定性指数), fomc_sentiment (FOMC文本情绪得分)
    触发: usepuindxd 20日Z-Score < -1.5 且开始起跳 (diff > 0) AND fomc_sentiment 60日边际变化 >= 0 -> +1.0
          usepuindxd 20日Z-Score > 1.5 且开始回落 (diff < 0) AND fomc_sentiment 60日边际变化 <= 0 -> -1.0
    输出: 狙击手脉冲信号, [-1.0, 1.0], 只在情绪极值反转的瞬间触发
    """

    def __init__(self):
        self.name = 'unstructured_policy_uncertainty_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化常态休眠信号为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查是否包含本因子需要的专属卫星数据
        if 'usepuindxd' not in data.columns or 'fomc_sentiment' not in data.columns:
            signal.name = self.name
            return signal
            
        epu = data['usepuindxd'].ffill()
        fomc = data['fomc_sentiment'].ffill()
        
        # 1. 不确定性指标极值评估 (20个交易日约为1个宏观月度观测窗口)
        epu_mean = epu.rolling(window=20).mean()
        epu_std = epu.rolling(window=20).std()
        epu_z = (epu - epu_mean) / epu_std.replace(0, np.nan)
        
        # 铁律2: 二阶导数 (边际反转，防止高位接飞刀)
        epu_diff = epu.diff()
        
        # 2. 联储文本情绪边际动量 
        # 铁律3: 边际变化。文本情绪是阶梯数据, 利用 .diff() 获取边际变化
        # 滚动 60个交易日 (约1个季度/2次议息会议) 求和，捕捉近期的宏观政策演变基调
        # > 0 表示基调边际转鸽，< 0 表示基调边际转鹰
        fomc_momentum = fomc.diff().rolling(window=60).sum()
        
        # 3. 非线性特征交叉
        # 做多条件: 自满被打破 (仍处低位区域但突然向上跳增) + 联储近期不鹰 (避险资金涌入)
        bull_cond = (epu_z < -1.5) & (epu_diff > 0) & (fomc_momentum >= 0)
        
        # 做空条件: 恐慌见顶衰竭 (极高不确定性开始实质性回落) + 联储近期不鸽 (避险资金撤退)
        bear_cond = (epu_z > 1.5) & (epu_diff < 0) & (fomc_momentum <= 0)
        
        # 触发脉冲信号
        signal[bull_cond] = 1.0
        signal[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"