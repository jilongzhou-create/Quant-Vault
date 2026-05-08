import numpy as np
import pandas as pd

class UnstructuredPolicyUncertaintyPulseFactor:
    """政策不确定性突变与衰竭脉冲因子 (policy_pivot / unstructured)

    逻辑: 基于每日经济政策不确定性(EPU)的新闻级边际变化。SPY作为均值回归的长牛资产：
          1. 抄底(极端恐慌+衰竭)：当政策不确定性长期处于极端高位(Z-score > 1.5)，且边际上连续两日显著回落时，意味着"利空出尽、政策路径明朗"，引发市场如释重负的强力反弹，触发+1.0脉冲。
          2. 趋势恶化(平静期轻微恐慌)：当市场处于常态平静期(-1.0 < Z < 1.0)时，政策不确定性突然大幅跳升(跳涨幅度超过1.5倍日常波动率)，打破了市场原有的平稳预期，引发杀估值，触发-1.0脉冲。
    数据: [usepuindxd] (Daily News Implied Economic Policy Uncertainty)
    输出: [-1.0, 1.0] 狙击手级别的非连续脉冲信号
    触发条件: 满足极端高位衰退或平静期意外暴涨，预期 Trigger Rate 控制在 8%~12% 之间。
    """

    def __init__(self):
        self.name = 'unstructured_policy_uncertainty_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全零的 Series 满足零值休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 检查数据列是否存在，缺失则返回全0
        if 'usepuindxd' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        
        # 计算 EPU 的10日均值，用于定义所处的不确定性"宏观状态(Regime)"
        epu_ma10 = epu.rolling(window=10, min_periods=10).mean()
        
        # 计算过去一年的 Regime 基准水位与波动率
        epu_roll_mean = epu_ma10.rolling(window=252, min_periods=252).mean()
        epu_roll_std = epu_ma10.rolling(window=252, min_periods=252).std()
        
        # 当前宏观状态的 Z-score (反映绝对水位的极值程度)
        epu_z = (epu_ma10 - epu_roll_mean) / (epu_roll_std + 1e-6)
        
        # 计算 EPU 的短期边际动量 (2日变化)，捕捉瞬时脉冲
        epu_diff2 = epu.diff(2)
        
        # 计算 EPU 变化的动态标准差，作为判定"突发脉冲幅度"的基准标尺
        epu_daily_std = epu.diff(1).rolling(window=252, min_periods=252).std()
        
        # 条件1：抄底买入 (+1.0)
        # 严格遵守"二阶导数铁律": 前一日必须处于极端恐慌状态 (Z > 1.5)，且今日开始大幅衰竭下行
        buy_cond = (epu_z.shift(1) > 1.5) & (epu_diff2 < -1.0 * epu_daily_std.shift(1))
        
        # 条件2：看空避险 (-1.0)
        # SPY物理属性: 平静期突发的不确定性飙升代表趋势恶化
        sell_cond = (epu_z.shift(1) > -1.0) & (epu_z.shift(1) < 1.0) & (epu_diff2 > 1.5 * epu_daily_std.shift(1))
        
        # 生成脉冲信号
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"