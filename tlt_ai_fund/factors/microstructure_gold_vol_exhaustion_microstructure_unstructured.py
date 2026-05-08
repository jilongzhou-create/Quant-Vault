import numpy as np
import pandas as pd

class UnstructuredEpuRegimePulseFactor:
    """Unstructured EPU Regime Pulse (Microstructure / Unstructured)

    逻辑: 美债(TLT)作为终极避险资产, 与经济政策不确定性(EPU)呈非线性相关. 
         1. 常态区间 (Z <= 1.5): EPU飙升引发避险情绪, 资金流入美债(买入); EPU回落则风险偏好升温, 资金流出(卖出).
         2. 极端恐慌 (Z > 1.5): 当不确定性达到极值且继续加速飙升时, 市场爆发流动性危机(如2020年3月), 投资者抛售一切资产换取现金, 美债亦遭无差别抛售(卖出). 只有当EPU见顶回落、恐慌衰竭时, 流动性修复, 避险资金才会真正涌入美债(买入).
         通过分离正常避险与极端流动性枯竭两种状态, 彻底解决直接抄底飞刀导致 CondIC 为负的问题.
    数据: usepuindxd (每日经济政策不确定性指数, 基于新闻文本的非结构化数据)
    触发: 
      - 常态避险(Z <= 1.5): EPU向上突破3日均线且单日跳跃 > 0.6σ -> 脉冲 +1.0; 向下突破且单日下跌 < -0.6σ -> 脉冲 -1.0
      - 极端反转(Z > 1.5): EPU极值衰竭(向下突破且单日大跌) -> 脉冲 +1.0; 极值加速(向上突破且暴涨, 流动性枯竭) -> 脉冲 -1.0
    输出: [-1.0, 1.0] 的极短期脉冲信号, 严格遵循零值休眠与二阶导数衰竭铁律.
    """

    def __init__(self):
        self.name = 'unstructured_epu_regime_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 如果缺少所需核心字段，返回全 0
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        epu = data['usepuindxd'].ffill()
        
        # 1. 计算宏观基准水位线 (252天滚动窗口)
        epu_mean = epu.rolling(window=252, min_periods=60).mean()
        epu_std = epu.rolling(window=252, min_periods=60).std()
        
        # 防止除以0
        epu_std = epu_std.replace(0, np.nan)
        epu_z = (epu - epu_mean) / epu_std
        
        # 2. 边际变化 (边际变化铁律: 使用 diff 和短均线捕捉瞬间动量)
        epu_diff = epu.diff(1)
        epu_ma3 = epu.rolling(window=3, min_periods=1).mean()
        
        # 3. 交叉信号捕捉反转瞬间 (二阶导数铁律)
        cross_up = (epu > epu_ma3) & (epu.shift(1) <= epu_ma3.shift(1))
        cross_down = (epu < epu_ma3) & (epu.shift(1) >= epu_ma3.shift(1))
        
        # 4. 狙击手脉冲过滤 (零值休眠铁律: 必须伴随大级别跳跃才触发, 保证Trigger Rate 5-15%)
        jump_up = cross_up & (epu_diff > 0.6 * epu_std)
        drop_down = cross_down & (epu_diff < -0.6 * epu_std)
        
        # 5. 划分宏观非线性状态区间
        extreme_panic = epu_z > 1.5
        normal_regime = epu_z <= 1.5
        
        # 6. 初始化信号为全 0.0 (休眠状态)
        signal = pd.Series(0.0, index=data.index)
        
        # --- 情景 A: 常态机制 (正相关避险) ---
        # 不确定性突增 -> 风险厌恶 -> 资金买入美债避险
        signal.loc[normal_regime & jump_up] = 1.0
        # 不确定性衰竭 -> 风险偏好恢复 -> 资金流出美债
        signal.loc[normal_regime & drop_down] = -1.0
        
        # --- 情景 B: 极端恐慌机制 (负相关流动性危机) ---
        # 极值状态下继续飙升 -> 流动性枯竭(Cash is King) -> 无差别抛售美债
        signal.loc[extreme_panic & jump_up] = -1.0
        # 极值状态见顶回落 -> 流动性危机解除, 终极避险属性回归 -> 抄底买入美债
        signal.loc[extreme_panic & drop_down] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"