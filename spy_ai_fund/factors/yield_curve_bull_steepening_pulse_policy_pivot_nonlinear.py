import numpy as np
import pandas as pd

class GoldilocksYieldPulsePolicyPivotNonlinearFactor:
    """Goldilocks Yield Pulse (policy_pivot/nonlinear)

    逻辑: 捕捉美联储政策预期突变导致的极短端收益率和曲线形态剧变。
         当短端利率(dgs2)急剧下降且曲线变陡，同时市场恐慌情绪(vix)平稳或下降时，意味着市场在为“软着陆降息”(Goldilocks)定价，强烈看多；
         反之，当短端急剧飙升、曲线走平且伴随恐慌上升时，为“鹰派紧缩冲击”，强烈看空。
         包含严格的防接飞刀过滤：如果在短端利率暴跌时VIX同样飙升，说明是因衰退恐慌导致的“倒逼降息”(如2008/2020)，该情况将被自动剔除。
    数据: dgs2, t10y2y, vixcls
    输出: +1.0 看多美股 (鸽派软着陆), -1.0 看空美股 (鹰派恐慌), 0.0 (休眠常态)
    触发条件: 5天动量的Z-Score极值交叉触发，预期Trigger Rate在 8%-15% 之间。
    """

    def __init__(self):
        self.name = 'goldilocks_yield_pulse_policy_pivot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['dgs2', 't10y2y', 'vixcls']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        # 1. 提取并清洗所需数据，向前填充避免低频数据导致的NaN
        df = data[required_cols].ffill()
        
        # 2. 计算边际动量变化 (窗口=5，完美覆盖FOMC/非农/CPI的一周核心发酵期)
        w = 5
        dgs2_chg = df['dgs2'].diff(w)
        t10y2y_chg = df['t10y2y'].diff(w)
        vix_chg = df['vixcls'].diff(w)
        
        # 3. 动态自适应Z-Score计算 (窗口=252，适应不同历史通胀与利率周期的绝对水位差异)
        z_w = 252
        min_p = 63
        epsilon = 1e-8  # 防止除以0
        
        dgs2_z = (dgs2_chg - dgs2_chg.rolling(window=z_w, min_periods=min_p).mean()) / \
                 (dgs2_chg.rolling(window=z_w, min_periods=min_p).std() + epsilon)
                 
        t10y2y_z = (t10y2y_chg - t10y2y_chg.rolling(window=z_w, min_periods=min_p).mean()) / \
                   (t10y2y_chg.rolling(window=z_w, min_periods=min_p).std() + epsilon)
                   
        vix_z = (vix_chg - vix_chg.rolling(window=z_w, min_periods=min_p).mean()) / \
                (vix_chg.rolling(window=z_w, min_periods=min_p).std() + epsilon)
                
        # 4. 基于经济学逻辑的非线性极值交叉触发
        
        # 鸽派脉冲 (Bullish - 软着陆/Goldilocks降息):
        # 1. 短端利率呈现1倍标准差级别的暴跌 (市场抢跑降息)
        # 2. 收益率曲线呈现 Bull Steepening (变陡)
        # 3. 二阶导数防御铁律：VIX并没有出现大于0.5个标准差的恐慌飙升(证明市场没在交易衰退/硬着陆)
        bull_cond = (dgs2_z < -1.0) & (t10y2y_z > 0.5) & (vix_z < 0.5)
        
        # 鹰派冲击 (Bearish - 流动性紧缩恐慌):
        # 1. 短端利率呈现1倍标准差级别的急剧上升 (加息预期升温)
        # 2. 收益率曲线 Bear Flattening (走平)
        # 3. 条件确认：VIX同步处于上升通道(Z>0.0)，证明此次加息预期真实刺穿了市场的风险偏好
        bear_cond = (dgs2_z > 1.0) & (t10y2y_z < -0.5) & (vix_z > 0.0)
        
        # 5. 脉冲信号写入
        signal.loc[bull_cond] = 1.0
        signal.loc[bear_cond] = -1.0
        
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"