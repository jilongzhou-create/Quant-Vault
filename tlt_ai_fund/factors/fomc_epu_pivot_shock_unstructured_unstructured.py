import numpy as np
import pandas as pd

class EpuShockDriftFactor:
    """EPU Uncertainty Shock Drift (Unstructured NLP)

    逻辑: 经济政策不确定性(EPU)急剧飙升通常表现为突发性恐慌(如地缘冲突、银行业危机)，驱动避险资金猛烈流入美债。
          为严格遵守"反接飞刀"铁律，因子只在EPU边际突变(Z-Score>1.5)且其二阶导数(加速度)开始衰竭时，才生成做多美债(+1.0)的跟随脉冲。
          反之，当不确定性断崖式下降且衰竭时，市场风险偏好回归，避险资金流出，做空美债(-1.0)。
    数据: usepuindxd (Daily Economic Policy Uncertainty Index)
    触发: 5日政策不确定性动量的252日Z-Score极值 (|Z| > 1.5) + 二阶动量反转(加速度转负/正) + 绝对水位确认
    输出: +1.0 (恐慌确立，避险流入); -1.0 (自满确立，避险流出)
    """

    def __init__(self):
        self.name = 'epu_shock_drift_unstructured'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 常态下信号必须严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 确保必需的无结构NLP数据字段存在
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 基础数据前向填充处理缺失值
        epu = data['usepuindxd'].ffill()
        
        # 1. 基础平滑：滤除单日新闻的高频噪音，提取连贯的微观政策不确定性波段 (3日均值)
        epu_smooth = epu.rolling(window=3, min_periods=1).mean()
        
        # 2. 铁律3 (边际变化)：计算 5 个交易日的变动量，捕捉短期的政策预期突变冲击
        epu_mom = epu_smooth.diff(5)
        
        # 3. 标准化评估：计算 252 日(一年)滚动 Z-Score，识别极端宏观突发事件
        roll_mean = epu_mom.rolling(window=252, min_periods=60).mean()
        roll_std = epu_mom.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        
        z_score = (epu_mom - roll_mean) / roll_std
        
        # 4. 铁律2 (二阶导数)：计算动量的变化率(加速度)，用于捕捉"突变极值 + 衰竭"
        epu_accel = epu_mom.diff(1)
        
        # 5. 绝对水位过滤：基准线(20日/一个月)确认当前不确定性确实处于非常态区间
        epu_ma20 = epu.rolling(window=20, min_periods=5).mean()
        
        # -------------------------------------------------------------
        # 多头信号 (+1.0) - 避险情绪爆发并沉淀:
        # 条件1: z_score > 1.5 : 不确定性发生极端的向上边际突变（恐慌发生）
        # 条件2: epu_accel < 0 : 恐慌的最快爆发阶段已过，趋势开始沉淀，避免接飞刀买在震荡最高点
        # 条件3: epu > epu_ma20: 绝对水位高于月均线，确认高波区
        # -------------------------------------------------------------
        long_cond = (z_score > 1.5) & (epu_accel < 0) & (epu > epu_ma20)
        
        # -------------------------------------------------------------
        # 空头信号 (-1.0) - 风险偏好回归并沉淀:
        # 条件1: z_score < -1.5 : 不确定性发生极端的向下边际突变（利好落地，恐慌骤降）
        # 条件2: epu_accel > 0  : 情绪释放的最快阶段已过，确认风险偏好结构性回归
        # 条件3: epu < epu_ma20 : 绝对水位低于月均线，确认低波区
        # -------------------------------------------------------------
        short_cond = (z_score < -1.5) & (epu_accel > 0) & (epu < epu_ma20)
        
        # 生成狙击手级别的脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"