import numpy as np
import pandas as pd

class UnstructuredPolicyPivotResonanceFactor:
    """Unstructured Policy Pivot Resonance Factor (unstructured/nonlinear)

    逻辑: 捕捉非结构化美联储文本态度突变(fomc_sentiment)与前端利率(dgs2)、利差(t10y2y)极值衰竭的非线性共振。依据三大铁律，绝对禁止使用低频文本情绪得分或持续倒挂状态的绝对值！多头脉冲仅在短端利率处于极值高位且开始暴跌(高位极值+急剧衰竭)，同时收益率曲线急剧变陡(动量陡峭确认降息定价)，或 FOMC 文本情绪发生剧烈鸽派突变(边际变化 Z > 2.5)时触发。这能有效滤除主跌浪中的噪音，精确狙击政策预期反转瞬间。
    数据: fomc_sentiment, dgs2, t10y2y
    触发: (dgs2 水位 Z > 1.5 且 dgs2.diff(5) Z < -2.0 且 t10y2y.diff(5) Z > 1.5) 或 fomc_sentiment.diff(5) Z > 2.5。
    输出: +1.0(看多美债脉冲) 或 -1.0(看空美债脉冲)，非触发日严格为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_policy_pivot_resonance_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['fomc_sentiment', 'dgs2', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # 填充缺失值防止计算中产生不必要的 NaN
        fomc = data['fomc_sentiment'].ffill()
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 1. 政策情绪边际突变 (Unstructured text-derived data)
        # 铁律3: 边际变化，绝不使用阶梯状 fomc_sentiment 的绝对水位
        # 使用 5 日差分捕捉会议前后的突变，以 252 日（一年）滚动计算 Z-Score
        fomc_diff = fomc.diff(5)
        fomc_z = (fomc_diff - fomc_diff.rolling(252).mean()) / (fomc_diff.rolling(252).std() + 1e-5)
        
        # 2. 短端利率极值衰竭 (Anti-Catch-Falling-Knife)
        # 条件1: 水位极值 (过去一年相对高点/低点)
        dgs2_level_z = (dgs2 - dgs2.rolling(252).mean()) / (dgs2.rolling(252).std() + 1e-5)
        # 条件2: 动量急剧回落/反抽 (季度窗口下的一阶导数极值)
        dgs2_diff = dgs2.diff(5)
        dgs2_diff_z = (dgs2_diff - dgs2_diff.rolling(63).mean()) / (dgs2_diff.rolling(63).std() + 1e-5)
        
        # 3. 收益率曲线动量变陡/变平确认 (Marginal Change Confirmation)
        # 过滤掉长期倒挂的绝对值状态，只关注"突然变陡(Bull Steepening)"的动量
        curve_diff = t10y2y.diff(5)
        curve_z = (curve_diff - curve_diff.rolling(63).mean()) / (curve_diff.rolling(63).std() + 1e-5)
        
        # --- 多头脉冲信号 (Bullish TLT) ---
        # 触发A: 纯粹的文本情绪鸽派突变 (Z-Score > 2.5)
        bull_shock = fomc_z > 2.5
        # 触发B: 短端利率见顶暴跌 (极值 Z>1.5 + 动量衰竭 Z<-2.0) + 曲线牛陡确认 (Z>1.5)
        bull_exhaustion = (dgs2_level_z > 1.5) & (dgs2_diff_z < -2.0) & (curve_z > 1.5)
        
        # --- 空头脉冲信号 (Bearish TLT) ---
        # 触发A: 纯粹的文本情绪鹰派突变 (Z-Score < -2.5)
        bear_shock = fomc_z < -2.5
        # 触发B: 短端利率见底飙升 (极值 Z<-1.5 + 动量爆发 Z>2.0) + 曲线熊平/倒挂加深确认 (Z<-1.5)
        bear_exhaustion = (dgs2_level_z < -1.5) & (dgs2_diff_z > 2.0) & (curve_z < -1.5)
        
        # 信号赋值
        signal[bull_shock | bull_exhaustion] = 1.0
        signal[bear_shock | bear_exhaustion] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"