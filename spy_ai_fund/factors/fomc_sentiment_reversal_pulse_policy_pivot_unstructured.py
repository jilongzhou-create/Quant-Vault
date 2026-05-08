import numpy as np
import pandas as pd

class EpuExhaustionPulseFactor:
    """新闻政策不确定性反转脉冲因子 (policy_pivot/unstructured)

    逻辑: 采用基于新闻文本构建的经济政策不确定性每日指数(EPU)。
          当EPU飙升至历史极端高位(Z>1.5)并开始拐头向下时(恐慌衰竭)，政策不确定性落地，
          美股通常迎来强劲的救市或风险偏好修复反弹(输出+1.0)；
          反之，当EPU处于历史低位(Z<-1.2，市场极度自满)且边际开始向上发酵时，
          预示新一轮政策黑天鹅风险正在酝酿，构成趋势恶化(输出-1.0)。
    数据: usepuindxd (Daily US Economic Policy Uncertainty Index)
    输出: 脉冲信号 [-1.0, 1.0]。
    触发条件: 1年期滚动Z-Score判断极值，叠加5日均线的边际动量反转。目标 Trigger Rate ~ 9%。
    """

    def __init__(self):
        self.name = 'epu_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查所需数据字段是否存在
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 提取数据并前向填充处理缺失值
        epu = data['usepuindxd'].ffill()
        
        # 1. 采用5个交易日(约一周)移动平均，过滤每日基于新闻文本提取时的高频噪音
        epu_ma = epu.rolling(window=5, min_periods=1).mean()
        
        # 2. 计算252个交易日(约1年)的滚动Z-Score，衡量当前政策不确定性的历史分位和极端程度
        epu_mean = epu_ma.rolling(window=252, min_periods=60).mean()
        epu_std = epu_ma.rolling(window=252, min_periods=60).std()
        
        # 防止除以0带来的无穷大异常
        epu_std = epu_std.replace(0, np.nan)
        epu_z = (epu_ma - epu_mean) / epu_std
        
        # 3. 边际变化：利用一阶导数判断不确定性演化的真实方向(动量)
        epu_mom = epu_ma.diff()
        
        # 初始化全零信号
        signal = pd.Series(0.0, index=data.index)
        
        # 4. 抄底买点: 极值+衰竭。政策不确定性处于极高位(恐慌)，但今日动量开始回落(风险落地/利空出尽)
        buy_cond = (epu_z > 1.5) & (epu_mom < 0.0)
        
        # 5. 看空卖点: 极低位+抬头。政策不确定性极低(市场自满)，但今日边际开始向上发酵(新风险酝酿)
        sell_cond = (epu_z < -1.2) & (epu_mom > 0.0)
        
        # 脉冲触发赋值
        signal.loc[buy_cond] = 1.0
        signal.loc[sell_cond] = -1.0
        
        # 命名并返回
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"