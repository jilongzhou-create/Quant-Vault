import numpy as np
import pandas as pd

class UnstructuredPolicyPivotSteepeningFactor:
    """政策预期突变与收益率曲线共振因子 (unstructured/nonlinear)

    逻辑: 捕捉美联储政策预期的极端跳跃 (fomc_sentiment 突变或 EPU 政策不确定性极值衰竭), 
          并通过对政策最敏感的短端利率(dgs2)与期限利差(t10y2y)的形态变化(Bull Steepening / Bear Flattening)进行非线性交叉验证。
          由于宏观政策预期的 Price-in 往往是瞬间脉冲，因子平时必须保持零值休眠，
          仅在预期突变且价格行为确认时输出脉冲信号，以规避连续单边接飞刀。
    数据: fomc_sentiment, usepuindxd, dgs2, t10y2y
    触发: 
      多头脉冲: (fomc_sentiment 5日变化 Z-Score > 2.5 OR (usepuindxd 变化 Z-Score > 2.5 且开始衰竭回落)) 
               AND (dgs2 下行 且 t10y2y 急剧变陡)
      空头脉冲: (fomc_sentiment 5日变化 Z-Score < -2.5 OR (usepuindxd 变化 Z-Score < -2.5 且开始反弹)) 
               AND (dgs2 上行 且 t10y2y 急剧变平)
    输出: +1.0 表示多重共振触发鸽派突变(看多美债)；-1.0 表示鹰派突变(看空美债)；常态休眠为 0.0。
    """

    def __init__(self, window=126):
        self.name = 'unstructured_policy_pivot_steepening_nonlinear'
        self.window = window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 基础数据校验
        required_cols = ['fomc_sentiment', 'usepuindxd', 'dgs2', 't10y2y']
        if not all(col in data.columns for col in required_cols):
            return pd.Series(0.0, index=data.index, name=self.name)

        df = data[required_cols].ffill()
        signal = pd.Series(0.0, index=df.index, name=self.name)

        # ==========================================
        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝对禁止使用低频数据绝对值，必须计算动量变化
        # ==========================================
        
        # Unstructured 1: FOMC Sentiment 鸽派/鹰派预期突变
        fomc_diff = df['fomc_sentiment'].diff(5)
        # 替换 std 为 0 的情况，防止除以零出现 inf
        fomc_std = fomc_diff.rolling(self.window).std().replace(0, np.nan)
        fomc_z = (fomc_diff - fomc_diff.rolling(self.window).mean()) / fomc_std
        
        # Unstructured 2: EPU (经济政策不确定性) 脉冲变化
        epu = df['usepuindxd']
        epu_diff3 = epu.diff(3)
        epu_z = (epu_diff3 - epu_diff3.rolling(self.window).mean()) / epu_diff3.rolling(self.window).std()

        # ==========================================
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 恐慌/极端指标必须加入衰竭与反转确认条件
        # ==========================================
        
        # EPU 恐慌极值且开始回落 (极端不确定性衰竭 -> 利好美债)
        epu_panic_exhaustion = (epu_z > 2.5) & (epu.diff(1) < 0)
        # EPU 极度自满且开始反弹 (平稳期被打破 -> 利空美债)
        epu_complacency_reversal = (epu_z < -2.5) & (epu.diff(1) > 0)

        # ==========================================
        # 方法C: 非线性特征交叉验证 (FICC 经济学逻辑)
        # 与收益率曲线的边际变化进行跨维度交叉验证
        # ==========================================
        
        dgs2 = df['dgs2']
        t10y2y = df['t10y2y']
        
        # Bull Steepening 确认: 短端利率(dgs2)急跌，且快于长端导致曲线变陡 (降息交易确认)
        dgs2_down = (dgs2.diff(5) < 0) & (dgs2.diff(1) < 0)
        curve_steepening = t10y2y.diff(5) > 0
        bull_steepening = dgs2_down & curve_steepening
        
        # Bear Flattening 确认: 短端利率(dgs2)急升，且快于长端导致曲线变平 (加息交易确认)
        dgs2_up = (dgs2.diff(5) > 0) & (dgs2.diff(1) > 0)
        curve_flattening = t10y2y.diff(5) < 0
        bear_flattening = dgs2_up & curve_flattening

        # ==========================================
        # 铁律1: 零值休眠 (Sniper Pulse)
        # ==========================================
        
        # 多头触发: (FOMC 鸽派突变 OR EPU 恐慌衰竭) 交叉叠加 (Bull Steepening)
        long_cond = ((fomc_z > 2.5).fillna(False) | epu_panic_exhaustion.fillna(False)) & bull_steepening.fillna(False)
        
        # 空头触发: (FOMC 鹰派突变 OR EPU 自满反转) 交叉叠加 (Bear Flattening)
        short_cond = ((fomc_z < -2.5).fillna(False) | epu_complacency_reversal.fillna(False)) & bear_flattening.fillna(False)

        # 仅在触发时赋极值，其余状态保持初始的 0.0
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        return signal

    def __repr__(self):
        return f"UnstructuredPolicyPivotSteepeningFactor(window={self.window})"