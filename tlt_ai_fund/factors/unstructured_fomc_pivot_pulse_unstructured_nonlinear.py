import numpy as np
import pandas as pd

class UnstructuredFomcPivotPulseFactor:
    """Unstructured FOMC Pivot & Exhaustion Factor (unstructured/nonlinear)

    逻辑: 捕捉 FOMC 鹰鸽情绪的阶梯状突变(脉冲)。通过 5日 diff 将低频绝对值转化为高频边际冲击。
          同时引入 EPU(经济政策不确定性) 的二阶导数作为衰竭过滤: 鸽派突变必须在宏观恐慌不再恶化时做多，
          鹰派突变必须在不确定性顺势升温时做空，绝对避免在单边主跌浪中逆势接飞刀。
    数据: fomc_sentiment (FOMC情绪得分), usepuindxd (经济政策不确定性)
    触发: FOMC情绪5日边际变化 Z-Score > 1.5 (鸽派)/ < -1.5 (鹰派) + EPU 3日动量出现衰竭/共振
    输出: +1.0 (鸽派突变看多), -1.0 (鹰派突变看空), 常态严格为 0.0 (狙击手脉冲)
    """

    def __init__(self):
        self.name = 'unstructured_fomc_pivot_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始信号全为 0.0
        signal = pd.Series(0.0, index=data.index)

        # 检查依赖列是否存在
        req_cols = ['fomc_sentiment', 'usepuindxd']
        for col in req_cols:
            if col not in data.columns:
                return signal

        # 前向填充缺失值
        fomc = data['fomc_sentiment'].ffill()
        epu = data['usepuindxd'].ffill()

        # --- 铁律3: 边际变化 (Marginal Change Only) ---
        # 绝对禁止使用 fomc_sentiment 的绝对水位！必须提取阶梯突变的脉冲
        fomc_mom = fomc.diff(5)
        
        # 计算 FOMC 边际变化的 252 日 Z-Score，识别极端转向冲击
        fomc_mom_mean = fomc_mom.rolling(window=252, min_periods=60).mean()
        fomc_mom_std = fomc_mom.rolling(window=252, min_periods=60).std()
        fomc_z = (fomc_mom - fomc_mom_mean) / (fomc_mom_std + 1e-8)

        # --- 铁律2: 二阶导数 (Anti-Catch-Falling-Knife) ---
        # 对高频噪波极大的 EPU 进行 3 日平滑
        epu_smooth = epu.rolling(window=3).mean()
        # 计算 EPU 的二阶导数特征，捕捉动量的衰竭与反转
        epu_diff = epu_smooth.diff(3)

        # 极端高水位过滤 (仅在具有一定宏观不确定性基础时，政策转向才具有强市价冲击)
        epu_z = (epu - epu.rolling(window=252, min_periods=60).mean()) / (epu.rolling(window=252, min_periods=60).std() + 1e-8)

        # 多头触发: FOMC发生超预期鸽派突变 (Z > 1.5) AND 宏观恐慌指数已经见顶开始回落 (二阶导 < 0) AND 当前存在一定避险需求 (EPU水位不过度低)
        long_cond = (fomc_z > 1.5) & (epu_diff < 0) & (epu_z > -1.0)

        # 空头触发: FOMC发生超预期鹰派突变 (Z < -1.5) AND 宏观恐慌指数同步恶化 (二阶导 > 0)
        short_cond = (fomc_z < -1.5) & (epu_diff > 0)

        # 赋值非零脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"