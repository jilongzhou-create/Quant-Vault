import numpy as np
import pandas as pd

class UnstructuredEpuVixMomentumExhaustionFactor:
    """非结构化政策与期权波动率动量背离衰竭因子 (unstructured/options)

    逻辑: 捕捉非结构化宏观政策不确定性(EPU)与期权隐含波动率(VIX)的动量背离。当政策恐慌远超股市恐慌(如意外鹰派或财政冲击)且开始衰竭时，意味着债市紧缩预期见顶，为美债绝佳抄底点；当股市闪崩远超政策不确定性(纯金融避险)且开始衰竭时，避险资金流出，做空美债。
    数据: usepuindxd (经济政策不确定性), vixcls (VIX期权波动率)
    触发: 两者动量差值的 252日 Z-Score 达到极值 (> 2.5 或 < -2.5)，且叠加二阶衰竭条件 (向3日均值回归)
    输出: +1.0 (极度看多) / -1.0 (极度看空)，常态为 0.0 (狙击手级脉冲)
    """

    def __init__(self):
        self.name = 'unstructured_epu_vix_momentum_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖数据是否存在
        required_cols = ['usepuindxd', 'vixcls']
        if not all(col in data.columns for col in required_cols):
            signal.name = self.name
            return signal

        df = data[required_cols].ffill()
        
        # 铁律3: 边际变化优先。绝对禁止使用绝对水位，此处计算5日(周频)动量变化
        epu_mom = df['usepuindxd'].diff(5)
        vix_mom = df['vixcls'].diff(5)
        
        # 消除两者量纲差异，分别计算自身的年度 (252日) Z-Score
        epu_mom_std = epu_mom.rolling(252).std().replace(0, np.nan)
        vix_mom_std = vix_mom.rolling(252).std().replace(0, np.nan)
        
        epu_mom_z = (epu_mom - epu_mom.rolling(252).mean()) / epu_mom_std
        vix_mom_z = (vix_mom - vix_mom.rolling(252).mean()) / vix_mom_std
        
        # 构建核心背离指标：政策恐慌动量 减去 股市恐慌动量
        mom_spread = epu_mom_z - vix_mom_z
        
        # 计算背离价差的 Z-Score，寻找极其罕见的宏观跨域错位事件
        spread_std = mom_spread.rolling(252).std().replace(0, np.nan)
        spread_z = (mom_spread - mom_spread.rolling(252).mean()) / spread_std
        
        # 铁律2: 二阶导数(Anti-Catch-Falling-Knife)。必须加入衰竭条件
        # 使用 3日均值 作为短期动量衰竭的参照基准
        mom_spread_ma3 = mom_spread.rolling(3).mean()
        
        # 触发条件1：政策恐慌远超股市恐慌 (如意外加息/通胀飙升) -> 债市超跌。
        # 当极值发生 (>2.5) 且差值开始回落 (< 3日均值) 时，说明利空出尽，抄底美债
        long_cond = (spread_z > 2.5) & (mom_spread < mom_spread_ma3)
        
        # 触发条件2：股市恐慌远超政策恐慌 (如纯金融系统闪崩) -> 债市避险拥挤。
        # 当极值发生 (<-2.5) 且差值开始反转 (> 3日均值) 时，说明避险情绪退潮，做空美债
        short_cond = (spread_z < -2.5) & (mom_spread > mom_spread_ma3)
        
        # 铁律1: 狙击手脉冲。满足条件才输出非零信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        # 异常值清理及命名
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"