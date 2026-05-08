import numpy as np
import pandas as pd

class UnstructuredPolicyPanicExhaustionFactor:
    """Policy Panic Exhaustion (unstructured/nonlinear)

    逻辑: 将非结构化的经济政策不确定性(EPU)的边际突变与短端利率(dgs2)、期限利差(t10y2y)的剧烈变化进行非线性交叉，构建“鹰派/鸽派恐慌指数”。因子仅在恐慌极值(Z>2.0)且开始衰竭时触发脉冲信号，从而在债券遭遇流动性抛售或恐慌性买入衰竭后精准抄底/逃顶。
    数据: usepuindxd (政策不确定性), dgs2 (2年期国债收益率), t10y2y (10年-2年利差)
    触发: Hawkish/Dovish Panic Z-Score > 2.0 且当前值小于3日均值(衰竭条件)
    输出: +1.0 (鹰派恐慌见顶衰竭，看多美债) / -1.0 (鸽派恐慌见顶衰竭，看空美债) / 0.0 (常态休眠脉冲)
    """

    def __init__(self, z_threshold: float = 2.0, diff_days: int = 5, window: int = 126):
        self.name = 'unstructured_policy_panic_exhaustion'
        self.z_threshold = z_threshold
        self.diff_days = diff_days
        self.window = window
        self.min_periods = 63

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['usepuindxd', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        df = data[required_cols].ffill(limit=5)
        
        # 铁律3: 边际变化 (Marginal Change Only) - 绝对禁止使用水平值，计算5日窗口内的动量跳跃
        epu_diff = df['usepuindxd'].diff(self.diff_days)
        dgs2_diff = df['dgs2'].diff(self.diff_days)
        t10y2y_diff = df['t10y2y'].diff(self.diff_days)
        
        # 对边际变化进行滚动标准化
        epu_z = (epu_diff - epu_diff.rolling(self.window, min_periods=self.min_periods).mean()) / (epu_diff.rolling(self.window, min_periods=self.min_periods).std() + 1e-8)
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(self.window, min_periods=self.min_periods).mean()) / (dgs2_diff.rolling(self.window, min_periods=self.min_periods).std() + 1e-8)
        t10y2y_z = (t10y2y_diff - t10y2y_diff.rolling(self.window, min_periods=self.min_periods).mean()) / (t10y2y_diff.rolling(self.window, min_periods=self.min_periods).std() + 1e-8)
        
        # 挖掘方法: 非线性特征交叉 (Non-linear Feature Cross)
        # 鹰派恐慌组合：政策不确定性飙升(+) + 2年期短端骤升(+) + 曲线熊平(-) -> 导致TLT被抛售
        hawkish_shock = epu_z + dgs2_z - t10y2y_z
        
        # 鸽派恐慌组合：政策不确定性飙升(+) + 2年期短端骤降(-) + 曲线牛陡(+) -> 导致TLT被抢筹
        dovish_shock = epu_z - dgs2_z + t10y2y_z
        
        # 计算综合恐慌指数的极端偏离度
        hawkish_z = (hawkish_shock - hawkish_shock.rolling(self.window, min_periods=self.min_periods).mean()) / (hawkish_shock.rolling(self.window, min_periods=self.min_periods).std() + 1e-8)
        dovish_z = (dovish_shock - dovish_shock.rolling(self.window, min_periods=self.min_periods).mean()) / (dovish_shock.rolling(self.window, min_periods=self.min_periods).std() + 1e-8)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife) - 必须等待极端动能出现衰竭
        hawkish_exhaustion = hawkish_z < hawkish_z.rolling(3).mean()
        dovish_exhaustion = dovish_z < dovish_z.rolling(3).mean()
        
        # 触发逻辑：达到极值阈值 且 拐头回落
        long_condition = (hawkish_z > self.z_threshold) & hawkish_exhaustion
        short_condition = (dovish_z > self.z_threshold) & dovish_exhaustion
        
        # 铁律1: 零值休眠 (Sniper Pulse) - 仅在触发瞬间输出非零脉冲
        signal[long_condition] = 1.0
        signal[short_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_threshold={self.z_threshold}, diff_days={self.diff_days})"