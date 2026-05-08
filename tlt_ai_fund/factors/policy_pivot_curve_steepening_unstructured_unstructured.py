import numpy as np
import pandas as pd

class UnstructuredPolicyPivotShockFactor:
    """政策预期突变脉冲因子 (unstructured/unstructured)

    逻辑: 捕捉美联储政策预期(FOMC情绪)的边际突变，及前端利率(dgs2)与期限利差(t10y2y)计价极值。为避免接飞刀，必须等动能开始衰竭(单日反转或阶梯走平)时才触发脉冲，在宏观趋势确立后的首次微幅回调介入。
    数据: fomc_sentiment, dgs2, t10y2y
    触发: FOMC情绪Z-Score极值且单日走平，或 dgs2 5日Z-Score极值伴随曲线印证且单日动能反转。
    输出: +1.0(看多美债/鸽派确立微调)，-1.0(看空美债/鹰派确立微调)。
    """

    def __init__(self):
        self.name = 'unstructured_policy_pivot_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['fomc_sentiment', 'dgs2', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        fomc = data['fomc_sentiment'].ffill()
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # --- 1. 非结构化 FOMC 情绪跳跃捕捉 ---
        # 铁律3: 边际变化。使用 3日差分捕捉低频阶梯变化
        fomc_diff3 = fomc.diff(3)
        fomc_roll_mean = fomc_diff3.rolling(126, min_periods=21).mean()
        fomc_roll_std = fomc_diff3.rolling(126, min_periods=21).std()
        fomc_z = (fomc_diff3 - fomc_roll_mean) / (fomc_roll_std + 1e-6)
        fomc_z = fomc_z.fillna(0)
        
        # 铁律2: 二阶导数/衰竭。情绪发生极端跳跃(Z>2.0)，且跳跃日已过(单日diff==0，于T+2/T+3确认)
        fomc_long = (fomc_z > 2.0) & (fomc.diff(1) == 0.0) & (fomc_diff3 > 0)
        fomc_short = (fomc_z < -2.0) & (fomc.diff(1) == 0.0) & (fomc_diff3 < 0)
        
        # --- 2. 前端利率与期限利差的趋势极值与回调介入 ---
        # 铁律3: 边际变化。短端利率是对政策最敏感的指标，使用5日动量
        dgs2_diff5 = dgs2.diff(5)
        dgs2_roll_mean = dgs2_diff5.rolling(63, min_periods=21).mean()
        dgs2_roll_std = dgs2_diff5.rolling(63, min_periods=21).std()
        dgs2_z = (dgs2_diff5 - dgs2_roll_mean) / (dgs2_roll_std + 1e-6)
        dgs2_z = dgs2_z.fillna(0)
        
        t10y2y_diff5 = t10y2y.diff(5)
        
        # 降息预期脉冲 (+1.0): 
        # 极值: 2年期大幅下行 (Z < -1.5) 且 曲线变陡 (t10y2y_diff5 > 0)
        # 衰竭: dgs2 单日不再下跌 (>= 0)，即宏观鸽派(看多TLT)趋势确立后的首次微幅回调，优化入场HitRate
        yield_long = (dgs2_z < -1.5) & (t10y2y_diff5 > 0) & (dgs2.diff(1) >= 0)
        
        # 加息预期脉冲 (-1.0):
        # 极值: 2年期大幅上行 (Z > 1.5) 且 曲线变平 (t10y2y_diff5 < 0)
        # 衰竭: dgs2 单日不再上涨 (<= 0)，即宏观鹰派(看空TLT)趋势确立后的首次微幅反弹
        yield_short = (dgs2_z > 1.5) & (t10y2y_diff5 < 0) & (dgs2.diff(1) <= 0)
        
        # --- 3. 信号合并 ---
        # 铁律1: 零值休眠。常态下严格为0，只在触发极值+衰竭时生成脉冲信号
        long_cond = fomc_long | yield_long
        short_cond = fomc_short | yield_short
        
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        # 异常重叠情况清零（极端互斥指标发生震荡时）
        overlap = long_cond & short_cond
        signal[overlap] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"