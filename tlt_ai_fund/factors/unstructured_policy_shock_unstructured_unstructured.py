import numpy as np
import pandas as pd

class UnstructuredPolicyShockFactor:
    """经济政策与央行情绪突变脉冲因子 (unstructured/unstructured)

    逻辑: 整合纯基于 NLP 文本提取的宏观经济政策不确定性(EPU)与美联储声明鹰鸽情绪(FOMC Sentiment)。当不确定性恐慌飙升至极值并见顶回落, 或美联储发布超预期鸽派声明时, 市场往往发生趋势性降息交易, 此时输出做多美债(TLT)脉冲; 反之, 当政策从极度平静中突变或美联储超预期偏鹰时, 触发做空脉冲。因子纯粹聚焦非结构化 NLP 域, 满足狙击手级别的严苛触发。
    数据: usepuindxd (经济政策不确定性), fomc_sentiment (FOMC文本鹰鸽情绪得分)
    触发: EPU Z-Score > 3.0 且回落 (二阶衰竭), 或 FOMC 3日边际变化 Z-Score > 2.5 产生瞬间跳跃。
    输出: 狙击手脉冲信号, [-1.0, 1.0], 日常休眠为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_policy_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值休眠信号 (铁律1)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必须的数据字段
        has_epu = 'usepuindxd' in data.columns
        has_fomc = 'fomc_sentiment' in data.columns
        
        if not has_epu and not has_fomc:
            return signal

        # =================================================================
        # 逻辑一: 经济政策不确定性 (EPU) 的极值衰竭脉冲
        # =================================================================
        if has_epu:
            epu = data['usepuindxd'].ffill()
            
            # 计算 252日滚动 Z-Score 衡量经济事件恐慌与平静的相对极端水位
            epu_std = epu.rolling(252).std().replace(0, np.nan).bfill().fillna(1e-4)
            epu_z = (epu - epu.rolling(252).mean()) / epu_std
            
            # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
            # 看多条件: 不确定性恐慌极值 (Z > 3.0) + 开始衰竭 (落于近期均线下方)
            epu_extreme_high = epu_z > 3.0
            epu_exhaustion_down = epu < epu.rolling(5).mean()
            long_pulse_epu = epu_extreme_high & epu_exhaustion_down
            
            # 看空条件: 极度平静期 (Z < -2.5) + 被打破 (突破均线开始上升)
            epu_extreme_low = epu_z < -2.5
            epu_breakout_up = epu > epu.rolling(5).mean()
            short_pulse_epu = epu_extreme_low & epu_breakout_up
            
            # 为保证仅在拐点瞬间触发脉冲，引入当日必须发生边际确认
            epu_just_reversed_down = epu.diff() < 0
            epu_just_reversed_up = epu.diff() > 0
            
            # 写入脉冲
            signal[long_pulse_epu & epu_just_reversed_down] = 1.0
            signal[short_pulse_epu & epu_just_reversed_up] = -1.0

        # =================================================================
        # 逻辑二: FOMC 情绪得分的边际突变脉冲
        # =================================================================
        if has_fomc:
            fomc = data['fomc_sentiment'].ffill()
            
            # 铁律3: 边际变化 (阶梯状数据的边际跳跃)
            # 绝对禁止使用绝对值, 仅在非结构化文本预期发生跳跃的瞬间捕捉
            fomc_diff = fomc.diff(3)
            fomc_diff_std = fomc_diff.rolling(252).std().replace(0, np.nan).bfill().fillna(1e-4)
            fomc_diff_z = (fomc_diff - fomc_diff.rolling(252).mean()) / fomc_diff_std
            
            # 触发极值突变 (超预期鸽派/鹰派转变)
            fomc_super_dove = fomc_diff_z > 2.5
            fomc_super_hawk = fomc_diff_z < -2.5
            
            # 二阶导数限制: 跳跃后动量衰减 (保证只有在跳跃发生的第一时间触发1天脉冲, 防止持续输出信号)
            fomc_just_jumped_dove = (fomc_diff > 0) & (fomc_diff >= fomc_diff.rolling(3).max())
            fomc_just_jumped_hawk = (fomc_diff < 0) & (fomc_diff <= fomc_diff.rolling(3).min())
            
            # 写入脉冲 (跳跃发生时会覆盖之前可能同方向产生的任何 EPU 弱信号，以美联储政策定盘为优先)
            signal[fomc_super_dove & fomc_just_jumped_dove] = 1.0
            signal[fomc_super_hawk & fomc_just_jumped_hawk] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"