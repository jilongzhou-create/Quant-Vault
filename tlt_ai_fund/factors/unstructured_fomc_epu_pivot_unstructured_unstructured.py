import numpy as np
import pandas as pd

class UnstructuredFomcEpuPivotFactor:
    """非结构化政策预期与不确定性突变因子 (unstructured/unstructured)

    逻辑: 结合基于新闻提取的经济政策不确定性(EPU)与基于FOMC声明文本的NLP情绪得分。当政策不确定性极其高涨并开始衰竭，同时央行情绪在边际上转鸽时，触发强烈的看多美债避险抢筹脉冲；当不确定性极度低迷且开始反弹，同时央行情绪边际转鹰时，触发看空脉冲。因子严格遵循零值休眠与二阶导衰竭铁律。
    数据: usepuindxd (经济政策不确定性指数), fomc_sentiment (FOMC鹰鸽得分, 1.0=极鸽)
    触发: 
      - 看多脉冲: EPU近5日最大 Z-Score > 2.2 且 当天下穿3日均值(二阶衰竭) 且 近21日FOMC情绪边际向鸽派变化 (diff > 0)
      - 看空脉冲: EPU近5日最小 Z-Score < -1.5 且 当天上穿3日均值(二阶反转) 且 近21日FOMC情绪边际向鹰派变化 (diff < 0)
    输出: +1.0 (鸽派避险) / -1.0 (鹰派复苏), 常态为 0.0 的狙击手级脉冲
    """

    def __init__(self, epu_lookback=252, z_long=2.2, z_short=-1.5):
        self.name = 'unstructured_fomc_epu_pivot_unstructured'
        self.epu_lookback = epu_lookback
        self.z_long = z_long
        self.z_short = z_short

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0.0 的脉冲信号序列
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['usepuindxd', 'fomc_sentiment']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        epu = data['usepuindxd']
        fomc = data['fomc_sentiment']
        
        # 铁律3: 边际变化 - 绝对禁止使用阶梯数据的绝对值！
        # FOMC 每年约8次会议(每隔45天左右)。使用 diff(21) 能够精确捕捉会议后三周内的预期边际跳跃
        fomc_momentum = fomc.diff(21)
        
        # 计算 EPU 长期 Z-Score
        epu_mean = epu.rolling(window=self.epu_lookback, min_periods=60).mean()
        epu_std = epu.rolling(window=self.epu_lookback, min_periods=60).std()
        epu_std = epu_std.replace(0, np.nan)
        epu_z = (epu - epu_mean) / epu_std
        
        # 铁律2: 二阶导数 - 必须等极端恐慌/极度自满情绪见顶并开始衰竭回落时，才能触发动作
        epu_ma3 = epu.rolling(window=3).mean()
        
        # 极值条件 (近5日内曾达到水位极值)
        epu_high_extreme = epu_z.rolling(window=5).max() > self.z_long
        epu_low_extreme = epu_z.rolling(window=5).min() < self.z_short
        
        # 衰竭/动量反转条件 (当天指标差分为负且跌破短期均线)
        exhaustion_down = (epu.diff() < 0) & (epu < epu_ma3)
        reversal_up = (epu.diff() > 0) & (epu > epu_ma3)
        
        # 触发脉冲的联合逻辑: 恐慌极值 + 开始消退 + 央行边际放鸽 = 做多债券脉冲
        long_cond = epu_high_extreme & exhaustion_down & (fomc_momentum > 0)
        
        # 反向逻辑: 自满极值 + 不确定性初露抬头 + 央行边际转鹰 = 做空债券脉冲
        short_cond = epu_low_extreme & reversal_up & (fomc_momentum < 0)
        
        # 铁律1: 零值休眠 (仅在事件满足时脉冲击发)
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(epu_lookback={self.epu_lookback}, z_long={self.z_long}, z_short={self.z_short})"