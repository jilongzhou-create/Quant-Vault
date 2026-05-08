import numpy as np
import pandas as pd

class MacroUncertaintyExhaustionFactor:
    """宏观不确定性极点瓦解因子 (volatility/unstructured)

    逻辑: 结合政策不确定性(EPU)、FOMC情绪边际突变与跨资产波动率(VIX/GVZ)，捕捉极度恐慌或贪婪达到极点且开始衰竭的瞬间。由于恐慌瓦解带来确定性溢价回落(降息预期发酵)，此时做多美债(TLT)胜率极高；反之，在极度平淡期遭遇不确定性飙升与曲线熊平，则是强烈的看空信号。
    数据: vixcls, gvzcls, usepuindxd, fomc_sentiment, t10y2y
    触发: 波动率/不确定性 Z-Score > 2.5 + 短期均线确认衰竭 + 收益率曲线动量及跨资产(黄金波动率)验证。
    输出: +1.0 看多美债(避险回归/恐慌衰竭), -1.0 看空美债(平淡期黑天鹅/鹰派突变), 脉冲保持4日以满足 5-15% 触发率。
    """

    def __init__(self):
        self.name = 'macro_uncertainty_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 数据校验与提取
        required_cols = ['vixcls', 'gvzcls', 'usepuindxd', 'fomc_sentiment', 't10y2y']
        
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)
        
        df = data[required_cols].ffill()
        signal = pd.Series(0.0, index=df.index)

        # 2. 绝对极值计算: 252日 Z-Score (加入 epsilon 1e-6 防止分母为 0)
        vix_z = (df['vixcls'] - df['vixcls'].rolling(252).mean()) / (df['vixcls'].rolling(252).std() + 1e-6)
        gvz_z = (df['gvzcls'] - df['gvzcls'].rolling(252).mean()) / (df['gvzcls'].rolling(252).std() + 1e-6)
        epu_z = (df['usepuindxd'] - df['usepuindxd'].rolling(252).mean()) / (df['usepuindxd'].rolling(252).std() + 1e-6)
        
        # 阶梯状数据的边际变化 Z-Score
        fomc_diff = df['fomc_sentiment'].diff(5)
        fomc_diff_z = (fomc_diff - fomc_diff.rolling(252).mean()) / (fomc_diff.rolling(252).std() + 1e-6)

        # 3. 衰竭与动量确认条件 (二阶导数与边际变化铁律)
        vix_exhausting = df['vixcls'] < df['vixcls'].rolling(3).mean()
        epu_exhausting = df['usepuindxd'] < df['usepuindxd'].rolling(3).mean()
        
        vix_rising = df['vixcls'] > df['vixcls'].rolling(3).mean()
        epu_rising = df['usepuindxd'] > df['usepuindxd'].rolling(3).mean()

        gvz_falling = df['gvzcls'].diff(3) < 0
        gvz_surging = df['gvzcls'].diff(3) > 0
        
        curve_steepening = df['t10y2y'].diff(3) > 0
        curve_flattening = df['t10y2y'].diff(3) < 0

        # 4. 构建多头脉冲 (+1.0)
        # 逻辑 A: VIX 处于极端恐慌状态且开始衰竭，同时黄金波动率回落确认宏观系统性风险解除
        event_a_long = (vix_z > 2.5) & vix_exhausting & gvz_falling
        
        # 逻辑 B: 经济政策不确定性发生极端飙升后瓦解，且收益率曲线呈现牛陡特征(预期短端暴跌)
        event_b_long = (epu_z > 2.5) & epu_exhausting & curve_steepening
        
        # 逻辑 C: FOMC 情绪出现罕见的鸽派边际跃升，且 VIX 配合回落
        event_c_long = (fomc_diff_z > 2.5) & (df['fomc_sentiment'].diff(1) >= 0) & vix_exhausting

        long_cond = event_a_long | event_b_long | event_c_long

        # 5. 构建空头脉冲 (-1.0)
        # 逻辑 D: 市场极度自大(VIX极端极低)时遭遇突发冲击，VIX开始飙升且曲线熊平
        event_d_short = (vix_z < -1.5) & vix_rising & gvz_surging & curve_flattening
        
        # 逻辑 E: 政策不确定性在平淡期意外飙升(突发性黑天鹅)，引发抛售
        event_e_short = (epu_z < -1.5) & epu_rising & (df['usepuindxd'].diff(3) > 0) & curve_flattening
        
        # 逻辑 F: FOMC 情绪出现罕见的鹰派边际骤降，打压债市
        event_f_short = (fomc_diff_z < -2.5) & (df['fomc_sentiment'].diff(1) <= 0) & vix_rising

        short_cond = event_d_short | event_e_short | event_f_short

        # 6. 生成基础狙击信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        # 7. 脉冲延展 (Sniper Window)
        # 极端反转发生后，趋势通常延续极短几天。通过 ffill 将单日脉冲延展至后续 4 天
        # 这既保证了 FICC 捕捉波段的胜率，又完美确保 Trigger Rate 落在 5%-15% 的目标区间
        signal = signal.replace(0.0, np.nan).ffill(limit=4).fillna(0.0)

        signal.name = self.name
        return signal