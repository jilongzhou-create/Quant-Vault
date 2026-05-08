import numpy as np
import pandas as pd

class EpuUncertaintyExhaustionFactor:
    """政策转向与不确定性脉冲 (policy_pivot/unstructured)

    逻辑: 捕捉基于新闻文本的经济政策不确定性(EPU)的变化。当政策不确定性极度飙升后开始回落时，标志着恐慌衰竭与政策预期落地，是看多美股的强烈抄底信号；当不确定性在平静期突然跃升(轻微恐慌发酵且尚未见顶)表明宏观风险恶化，是看空信号。
    数据: usepuindxd (Daily News-Based Economic Policy Uncertainty Index)
    输出: 衰竭买入为 +1.0，恐慌突升看空为 -1.0
    触发条件: 126日Z-Score > 2.0且从近期高点回落超12%触发看多(脉冲持续3天)；5天内激增超40%且Z-Score处于[1.0, 2.0]触发看空。预期 Trigger Rate 约 8%-12%。
    """

    def __init__(self):
        self.name = 'epu_uncertainty_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        epu = data['usepuindxd'].ffill()
        
        # 3日平滑，过滤单日基于新闻的极值噪音
        epu_smooth = epu.rolling(window=3).mean()
        
        # 126天（约半年）的滚动Z-score，适应局部宏观不确定性的基准Regime变化
        epu_mean = epu_smooth.rolling(window=126).mean()
        epu_std = epu_smooth.rolling(window=126).std()
        epu_z = (epu_smooth - epu_mean) / epu_std.replace(0, np.nan)
        
        # ==========================================
        # 看多信号 (极端不确定性 + 拐点衰竭买入)
        # ==========================================
        # 近10天内出现过极度的政策不确定性 (防接飞刀，必须确认之前存在恐慌)
        recent_extreme = epu_z.rolling(window=10).max() > 2.0
        
        # 计算近期高点回撤幅度
        epu_local_peak = epu_smooth.rolling(window=10).max()
        epu_drawdown = (epu_smooth / epu_local_peak.replace(0, np.nan)) - 1.0
        
        # 不确定性开始显著回落 (回落超 12% 且边际仍在下降)
        is_exhausted = (epu_drawdown < -0.12) & (epu_smooth.diff() < 0)
        
        bull_cond = recent_extreme & is_exhausted
        bull_pulse = bull_cond & ~bull_cond.shift(1).fillna(False)
        
        # ==========================================
        # 看空信号 (轻微恐慌跃升，风险开始发酵)
        # ==========================================
        # 短期不确定性动量激增
        epu_ret5 = (epu_smooth / epu_smooth.shift(5).replace(0, np.nan)) - 1.0
        
        # 未达到极度恐慌状态 (Z-Score < 2.0，极度恐慌往往对应主跌浪末端，不能在此看空)
        mild_high = (epu_z > 1.0) & (epu_z <= 2.0)
        
        bear_cond = (epu_ret5 > 0.40) & mild_high & (epu_smooth.diff() > 0)
        bear_pulse = bear_cond & ~bear_cond.shift(1).fillna(False)
        
        # 转换为3天有效期的脉冲，以维持 5%-15% 的 Trigger Rate 要求
        bull_signal = bull_pulse.rolling(window=3).max().fillna(0) > 0
        bear_signal = bear_pulse.rolling(window=3).max().fillna(0) > 0
        
        # 组装最终信号
        signal = pd.Series(0.0, index=data.index)
        signal.loc[bull_signal] = 1.0
        signal.loc[bear_signal] = -1.0
        
        # 极小概率下的冲突静默保护
        conflict = bull_signal & bear_signal
        signal.loc[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"