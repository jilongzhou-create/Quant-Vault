import numpy as np
import pandas as pd

class UnstructuredFomcShockExhaustionFactor:
    """Unstructured FOMC Shock Exhaustion (unstructured/options)

    逻辑: 基于 FOMC 新闻情绪得分的突变(边际跳跃)捕捉联储政策(鹰/鸽)转折。为防止在政策冲击当天的无序抛售中接飞刀，必须等待期权波动率(VIX)和政策不确定性(EPU)从高位边际回落(靴子落地衰竭确认)，此时顺政策方向强力切入美债，形成精准的高胜率脉冲。
    数据: fomc_sentiment (非结构化新闻情绪), vixcls (期权波动率), usepuindxd (经济政策不确定性)
    触发: fomc_sentiment 3日边际变化 Z-Score > 2.5 (鸽派突变) 或 < -2.5 (鹰派突变) AND 波动率/不确定性边际衰竭 (当日小于3日均值)。
    输出: +1.0 看多美债(政策超预期转鸽且恐慌消退), -1.0 看空美债(政策超预期转鹰且恐慌消退)。其余常态时间严格为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_shock_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 初始信号设为全 0.0，严格遵守零值休眠铁律 (狙击手模式)
        signal = pd.Series(0.0, index=data.index)
        
        # 数据存在性检查
        required_cols = ['fomc_sentiment', 'vixcls', 'usepuindxd']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 2. 获取所需数据并处理缺失值 (严禁引用 CoreAnchor 数据)
        fomc = data['fomc_sentiment'].ffill()
        vix = data['vixcls'].ffill()
        epu = data['usepuindxd'].ffill()
        
        # 3. 边际变化铁律：绝对禁止使用 fomc_sentiment 绝对值！只捕捉预期改变的瞬间！
        # 使用 3 日差分捕捉低频阶梯数据的边际动能跳跃
        fomc_mom = fomc.diff(3).fillna(0.0)
        
        # 计算动能跳跃的 252 日滚动 Z-Score (一年为基准评估当前的政策冲击烈度)
        fomc_mom_mean = fomc_mom.rolling(window=252, min_periods=21).mean()
        fomc_mom_std = fomc_mom.rolling(window=252, min_periods=21).std()
        
        # 稳健性处理：防止由于大部分时间 diff 为 0 导致的标准差极小/缺失
        fomc_mom_std = fomc_mom_std.replace(0.0, 1e-5).fillna(1e-5)
        fomc_z = (fomc_mom - fomc_mom_mean) / fomc_mom_std
        
        # 4. 二阶导数铁律：必须有靴子落地的恐慌衰竭确认，绝不在波动最高峰接飞刀！
        # 监测 VIX 和 EPU 是否开始回落 (当日值向下击穿过去3日均值，证明单边恐慌/抛压已过巅峰)
        vix_exhaustion = vix < vix.rolling(window=3).mean()
        epu_exhaustion = epu < epu.rolling(window=3).mean()
        
        # 只要微观期权波动率(VIX)或宏观政策不确定性(EPU)其一出现明显降温，即视为抛压衰竭
        shock_exhausted = vix_exhaustion | epu_exhaustion
        
        # 5. 脉冲信号发射：
        # 极度鸽派突变 + 恐慌边际衰竭 -> 强降息预期确立且抛压结束 -> 看多美债 (+1.0)
        bull_cond = (fomc_z > 2.5) & shock_exhausted
        
        # 极度鹰派突变 + 恐慌边际衰竭 -> 强加息预期确立且护盘结束 -> 看空美债 (-1.0)
        bear_cond = (fomc_z < -2.5) & shock_exhausted
        
        signal[bull_cond] = 1.0
        signal[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"