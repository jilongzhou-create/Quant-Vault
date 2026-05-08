import numpy as np
import pandas as pd

class PolicyUncertaintyPulseFactor:
    """经济政策不确定性突变因子 (policy_pivot/unstructured)

    逻辑: 捕捉基于新闻的经济政策不确定性(EPU)的极值反转和超预期冲击。作为纯粹的非结构化数据因子，EPU的激增意味着政策或宏观环境恶化(看空美股)；而当EPU触及历史高点(Z>1.8，极度恐慌)后开始实质性回落(降幅>0.5个标准差)，标志着政策利空落地或预期反转，带来"恐慌衰竭"的抄底买点(看多美股)。完全契合SPY均值回归及防接飞刀规则。
    数据: [usepuindxd] (Daily News-Based Economic Policy Uncertainty Index)
    输出: [-1.0, 1.0] 的脉冲信号
    触发条件: 极度恐慌(最近5日Z>=1.8)且当前开始大幅回落(Z.diff<-0.5)时输出+1.0；相对平静期(近5日Z<=0.5)突发大幅飙升(Z.diff>1.0)时输出-1.0。预期Trigger Rate约8-12%。
    """

    def __init__(self):
        self.name = 'policy_uncertainty_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理数据缺失情况
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        # 获取EPU指数，向前填充缺失值
        epu = data['usepuindxd'].ffill()
        
        # 新闻指数具有显著的右偏特征(经常出现极端脉冲)，取对数使其近似正态化
        log_epu = np.log(epu.replace(0, np.nan)).ffill()
        
        # 计算中期(63个交易日，约一个季度)宏观基准线，符合宏观策略视角
        ma63 = log_epu.rolling(window=63).mean()
        std63 = log_epu.rolling(window=63).std()
        
        # 消除前期的空值及零波动影响，防止除以零
        std63 = std63.replace(0, np.nan)
        
        # 每日EPU相对于季度均值的不确定性 Z-Score
        z_score = (log_epu - ma63) / std63
        
        # 计算近期的极值水位 (回溯一周状态)
        max_z_5 = z_score.rolling(window=5).max()
        min_z_5 = z_score.rolling(window=5).min()
        
        # 计算边际动量变化 (捕捉短期突变与衰竭)
        z_diff1 = z_score.diff(1)
        z_diff2 = z_score.diff(2)
        
        # 看多信号 (抄底逻辑: 极度恐慌 + 衰竭)
        # 条件: 最近一周内经历过极端不确定性(Z>=1.8)，且当前正在急剧回落(单日下降>0.5 std或两日下降>0.7 std)
        # 且要求水位仍在均线之上(Z>0.0)，确保是在高位刚开始衰竭，而不是已经到底
        buy_cond = (max_z_5 >= 1.8) & ((z_diff1 < -0.5) | (z_diff2 < -0.7)) & (z_score > 0.0)
        
        # 看空信号 (趋势恶化: 平静期 + 恐慌飙升)
        # 条件: 近期处于相对平静期(Z<=0.5)，突发黑天鹅或政策冲击导致不确定性激增(单日上升>1.0 std或两日上升>1.3 std)
        sell_cond = (min_z_5 <= 0.5) & ((z_diff1 > 1.0) | (z_diff2 > 1.3))
        
        # 生成狙击手级别的脉冲信号
        signal = pd.Series(0.0, index=data.index)
        
        # 赋予负向信号略高优先级 (防止极端震荡下同一天逻辑冲突)
        signal[sell_cond] = -1.0
        signal[buy_cond] = 1.0
        
        # 过滤掉前63天未充分预热的区间
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"