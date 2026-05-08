import numpy as np
import pandas as pd

class NewsPanicExhaustionFactor:
    """新闻恐慌极值与衰竭因子 (panic_mean_reversion/unstructured)

    逻辑: 使用每日新闻文本抽取的经济政策不确定性指数(usepuindxd)作为非结构化恐慌指标。
          当政策不确定性处于近一季度极高位(Z>1.5)，且当天出现边际回落时，标志着"恐慌极值+衰竭"，触发抄底买入脉冲。
          当政策不确定性处于常态偏高但未极值，且连续3天暗中升温(钝刀割肉)，标志着市场环境边际恶化，触发看空脉冲。
          (注: 放弃FOMC sentiment是因为其低频阶梯特性会导致Trigger Rate远低于5%下限，故采用同属非结构化新闻文本生成的usepuindxd)
    数据: usepuindxd (基于新闻报道的每日经济政策不确定性指数)
    输出: +1.0(强力看多，极端恐慌衰竭)，-1.0(看空，不确定性阴跌升温)
    触发条件: Z-Score > 1.5 且当天回落则+1；0 < Z < 1.5 且连续3天上升则-1。预期Trigger Rate ~8%
    """

    def __init__(self, window=63, z_extreme=1.5):
        self.name = 'news_panic_exhaustion_pulse'
        self.window = window
        self.z_extreme = z_extreme

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查所需数据是否在 DataFrame 中
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
        
        # 获取日常不确定性指数并前向填充缺失值
        epu = data['usepuindxd'].ffill()
        
        # 指数本身偏度大，取对数进行平滑处理以使标准差计算更稳健
        log_epu = np.log1p(epu)
        
        # 计算短周期(约一个季度)内的 Z-Score
        epu_mean = log_epu.rolling(window=self.window).mean()
        epu_std = log_epu.rolling(window=self.window).std()
        
        # 避免除以 0 的情况
        epu_z = (log_epu - epu_mean) / (epu_std + 1e-8)
        
        # 计算边际变化
        epu_diff = epu.diff(1)
        
        # ---------------------------------------------------------
        # 核心物理法则 1: 极值 + 衰竭 = 买入点 (抄底)
        # ---------------------------------------------------------
        # Z-Score > 1.5 表明新闻层面存在极其严重的恐慌和不确定性
        # epu_diff < 0 且当天不确定性低于近3日均值，表明恐慌开始退潮 (二阶导为负)
        long_condition = (
            (epu_z > self.z_extreme) & 
            (epu_diff < 0) & 
            (epu < epu.rolling(window=3).mean())
        )
        
        # ---------------------------------------------------------
        # 核心物理法则 2: 连续轻微升温 = 钝刀割肉卖点 (看空)
        # ---------------------------------------------------------
        # 0 < Z-Score <= 1.5 表明处于不确定性常态偏上方
        # 连续 3 天 diff > 0 表明新闻中的不安情绪在暗中发酵，趋势正在恶化
        up_3_days = (
            (epu_diff > 0) & 
            (epu.shift(1).diff(1) > 0) & 
            (epu.shift(2).diff(1) > 0)
        )
        short_condition = (
            (epu_z > 0.0) & 
            (epu_z <= self.z_extreme) & 
            up_3_days
        )
        
        # 生成脉冲信号，常态返回 0.0
        signal = pd.Series(0.0, index=data.index)
        signal[long_condition] = 1.0
        signal[short_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_extreme={self.z_extreme})"