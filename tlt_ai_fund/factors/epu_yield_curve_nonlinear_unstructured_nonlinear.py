import numpy as np
import pandas as pd

class EpuYieldCurveNonlinearFactor:
    """政策不确定性与收益率曲线交叉因子 (unstructured/nonlinear)

    逻辑: 经济政策不确定性(EPU)的极端脉冲往往预示着宏观范式的切换。当EPU短期剧烈飙升达到极值并开始边际衰竭时，如果伴随2年期美债收益率的边际下行及曲线的牛陡(Bull Steepening)，说明市场正在从极度恐慌中Price-in美联储的鸽派转向，此时买入美债(TLT)。反之，当EPU极度乐观并反弹，且收益率熊平，说明遭遇超预期的鹰派惊吓。
    数据: usepuindxd (经济政策不确定性), dgs2 (2年期收益率), t10y2y (期限利差)
    触发: EPU 5日变化量 Z-Score 绝对值 > 2.5 + EPU边际动量反转(二阶导数衰竭) + 收益率曲线动态的同步非线性印证
    输出: +1.0 看多美债, -1.0 看空美债, 常态为 0.0
    """

    def __init__(self, window=126, momentum_days=5):
        self.name = 'epu_yield_curve_nonlinear'
        self.window = window
        self.momentum_days = momentum_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        required_cols = ['usepuindxd', 'dgs2', 't10y2y']
        
        # 严格处理缺失列, 返回全0序列
        if not all(col in data.columns for col in required_cols):
            return pd.Series(0.0, index=data.index, name=self.name)
            
        df = data[required_cols].ffill()
        
        # 1. 边际变化铁律: 全部使用动量变化而非绝对水位, 捕捉预期突变的瞬间
        epu_chg = df['usepuindxd'].diff(self.momentum_days)
        dgs2_chg = df['dgs2'].diff(self.momentum_days)
        t10y2y_chg = df['t10y2y'].diff(self.momentum_days)
        
        # 2. 计算不确定性(EPU)边际变化的滚动 Z-Score 
        epu_chg_mean = epu_chg.rolling(self.window).mean()
        epu_chg_std = epu_chg.rolling(self.window).std()
        epu_z = (epu_chg - epu_chg_mean) / (epu_chg_std + 1e-8)
        
        # 二阶导数铁律: 动量的再变动 (用于判断极端情绪是否开始衰竭反转)
        epu_chg_diff = epu_chg.diff(1)
        
        # 初始化 Sniper Pulse 休眠信号
        signal = pd.Series(0.0, index=df.index)
        
        # 3. 多头条件 (鸽派突变 -> 看多美债 TLT):
        # - 条件A (极值): 政策不确定性极大爆发 (Z-score > 2.5)
        # - 条件B (衰竭): 不确定性动量开始回落 (epu_chg_diff < 0)
        # - 条件C (交叉印证): 短端收益率下行(降息预期, <-5bps) 且 曲线变陡(Bull Steepening, >+2bps)
        long_cond = (
            (epu_z > 2.5) & 
            (epu_chg_diff < 0) & 
            (dgs2_chg < -0.05) & 
            (t10y2y_chg > 0.02)
        )
        
        # 4. 空头条件 (鹰派惊吓 -> 看空美债 TLT):
        # - 条件A (极值): 政策不确定性异常收缩至冰点 (Z-score < -2.5)
        # - 条件B (衰竭): 极度自满情绪开始反弹/恶化 (epu_chg_diff > 0)
        # - 条件C (交叉印证): 短端收益率飙升(加息预期, >+5bps) 且 曲线变平(Bear Flattening, <-2bps)
        short_cond = (
            (epu_z < -2.5) & 
            (epu_chg_diff > 0) & 
            (dgs2_chg > 0.05) & 
            (t10y2y_chg < -0.02)
        )
        
        # 赋值脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, momentum_days={self.momentum_days})"