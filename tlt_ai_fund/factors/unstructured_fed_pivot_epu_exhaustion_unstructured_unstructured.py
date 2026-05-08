import numpy as np
import pandas as pd

class UnstructuredFedPivotEpuExhaustionFactor:
    """非结构化美联储转向与不确定性衰竭共振因子 (unstructured/unstructured)

    逻辑: 捕捉美联储FOMC非结构化情绪得分的极端边际变化。当FOMC政策发生出乎意料的鹰/鸽派突变(跳跃)，且伴随宏观政策不确定性(EPU)的冲高回落(利空/利多落地)时，触发短暂的顺势买卖脉冲。
    数据: fomc_sentiment (FOMC情绪得分), usepuindxd (经济政策不确定性指数)
    触发: 
      条件1: FOMC情绪10日边际变化的 252日Z-Score绝对值 > 1.5 (捕捉显著超预期跳跃)
      条件2: 情绪跳变动量开始衰竭 (当前Z-score相对昨日回落，避免主跌/主升浪直接接刀)
      条件3: 政策不确定性指数(EPU)跌破5日均线 (确认恐慌/亢奋情绪宣泄完毕)
    输出: +1.0 鸽派反转落地看多脉冲, -1.0 鹰派反转落地看空脉冲
    """

    def __init__(self, fomc_window=10, z_window=252, z_thresh=1.5, epu_window=5):
        self.name = 'unstructured_fed_pivot_epu_exhaustion_unstructured'
        self.fomc_window = fomc_window   # 边际变化窗口，覆盖会议前后的预期跳跃
        self.z_window = z_window         # 长期基准窗口(252个交易日约一年)
        self.z_thresh = z_thresh         # 极端脉冲阈值
        self.epu_window = epu_window     # 短期不确定性衰竭窗口

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 - 初始全0序列
        signal = pd.Series(0.0, index=data.index)
        
        if 'fomc_sentiment' not in data.columns or 'usepuindxd' not in data.columns:
            return signal
            
        fomc = data['fomc_sentiment'].ffill().fillna(0.0)
        epu = data['usepuindxd'].ffill().fillna(0.0)
        
        # 铁律3: 边际变化 - 绝对禁止使用阶梯数据的绝对值，使用滚动差分捕捉跳跃
        fomc_diff = fomc.diff(self.fomc_window)
        
        # 计算 FOMC边际变化 的 Z-Score 以判定跳跃的极端程度
        fomc_diff_mean = fomc_diff.rolling(window=self.z_window, min_periods=60).mean()
        fomc_diff_std = fomc_diff.rolling(window=self.z_window, min_periods=60).std()
        fomc_z = (fomc_diff - fomc_diff_mean) / (fomc_diff_std + 1e-8)
        
        # 铁律2: 二阶导数 - 极值必须伴随动量衰竭(防接飞刀)
        # 1. FOMC情绪冲击动量自身的拐点确认
        fomc_z_falling = fomc_z < fomc_z.shift(1)  # 鸽派冲击动量见顶回落
        fomc_z_rising = fomc_z > fomc_z.shift(1)   # 鹰派冲击动量见底回升
        
        # 2. 政策不确定性衰竭(落地确认：不确定性回落说明市场已经消化决议)
        epu_mean = epu.rolling(window=self.epu_window, min_periods=2).mean()
        epu_exhaustion = epu < epu_mean
        
        # 触发脉冲条件
        # 做多: 发生极端鸽派突变(Z > 1.5) + 鸽派动能停止强化 + 宏观不确定性宣泄完毕
        long_cond = (fomc_z > self.z_thresh) & fomc_z_falling & epu_exhaustion
        
        # 做空: 发生极端鹰派突变(Z < -1.5) + 鹰派动能停止强化 + 宏观不确定性宣泄完毕
        short_cond = (fomc_z < -self.z_thresh) & fomc_z_rising & epu_exhaustion
        
        # 赋值并清洗潜在的 NaN
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(fomc_window={self.fomc_window}, z_window={self.z_window}, z_thresh={self.z_thresh}, epu_window={self.epu_window})"