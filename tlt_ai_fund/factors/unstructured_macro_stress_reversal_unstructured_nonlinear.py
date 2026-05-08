import numpy as np
import pandas as pd

class UnstructuredMacroStressReversalFactor:
    """非结构化宏观情绪衰竭交叉曲线动量因子 (unstructured/nonlinear)

    逻辑: 结合经济政策不确定性(EPU)与VIX构建综合宏观压力指数。当非结构化压力极度恐慌且开始衰竭时，若同时伴随短端利率急剧下行导致的曲线陡峭化(联储转向降息预期)，则生成看多美债脉冲；当情绪极度自满且开始苏醒时，伴随短端飙升导致曲线平坦化(紧缩超预期)，则生成看空脉冲。这是高胜率的二阶导数反转脉冲，完全避免单边接飞刀。
    数据: usepuindxd, vixcls, dgs2, t10y2y
    触发: 压力综合 Z-Score > 2.0 且边际回落 + 短端利率下行 Z-Score < -1.5 且曲线急剧变陡 → +1.0 脉冲
    输出: [-1.0, 1.0] 的离散脉冲信号
    """

    def __init__(self, z_window=60, diff_window=5):
        self.name = 'unstructured_macro_stress_reversal'
        self.z_window = z_window         # 季度级基准参考窗口
        self.diff_window = diff_window   # 捕捉周级别边际预期的突变

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，默认全局0.0
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['usepuindxd', 'vixcls', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        df = data[required_cols].ffill()
        
        # 1. 构建综合宏观压力指数 (Unstructured EPU + VIX)
        # 铁律3: 针对状态变量提取标准化的动量特征
        epu_z = (df['usepuindxd'] - df['usepuindxd'].rolling(self.z_window).mean()) / (df['usepuindxd'].rolling(self.z_window).std() + 1e-6)
        vix_z = (df['vixcls'] - df['vixcls'].rolling(self.z_window).mean()) / (df['vixcls'].rolling(self.z_window).std() + 1e-6)
        
        stress_idx = epu_z + vix_z
        stress_z = (stress_idx - stress_idx.rolling(self.z_window).mean()) / (stress_idx.rolling(self.z_window).std() + 1e-6)
        
        # 2. 利率动量提取 (铁律3: 纯边际变化，捕捉预期跃升)
        dgs2_diff = df['dgs2'].diff(self.diff_window)
        t10y2y_diff = df['t10y2y'].diff(self.diff_window)
        
        dgs2_diff_z = (dgs2_diff - dgs2_diff.rolling(self.z_window).mean()) / (dgs2_diff.rolling(self.z_window).std() + 1e-6)
        t10y2y_diff_z = (t10y2y_diff - t10y2y_diff.rolling(self.z_window).mean()) / (t10y2y_diff.rolling(self.z_window).std() + 1e-6)
        
        # 3. 二阶导数与极值反转条件 (铁律2: 拒绝接飞刀，必须附带衰竭条件)
        
        # 多头条件：宏观恐慌极值 (>2.0) 且开始回落 + 降息预期骤升 (短端暴跌且仍在下行) + 曲线变陡 (Bull Steepening)
        panic_exhaustion = (stress_z > 2.0) & (stress_idx.diff(2) < 0)
        rate_cut_shock = (dgs2_diff_z < -1.5) & (df['dgs2'].diff(1) < 0) & (t10y2y_diff_z > 1.0)
        long_cond = panic_exhaustion & rate_cut_shock
        
        # 空头条件：宏观极度自满 (<-2.0) 且开始反弹 + 紧缩预期骤升 (短端暴涨且仍在上升) + 曲线变平/倒挂加深 (Bear Flattening)
        complacency_reversal = (stress_z < -2.0) & (stress_idx.diff(2) > 0)
        rate_hike_shock = (dgs2_diff_z > 1.5) & (df['dgs2'].diff(1) > 0) & (t10y2y_diff_z < -1.0)
        short_cond = complacency_reversal & rate_hike_shock
        
        # 触发狙击手脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"UnstructuredMacroStressReversalFactor(z_window={self.z_window}, diff_window={self.diff_window})"