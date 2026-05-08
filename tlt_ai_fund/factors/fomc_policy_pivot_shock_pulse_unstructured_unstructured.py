import numpy as np
import pandas as pd

class FomcPolicyPivotShockPulseFactor:
    """FOMC政策预期突变反转脉冲因子 (unstructured/nonlinear)

    逻辑: 捕捉美联储政策预期的极端跳跃和短端利率定价的快速变轨。对低频文本情绪（fomc_sentiment）通过一阶差分提取瞬时边际突跳抓取转向脉冲；同时对高频2年期美债（dgs2）和曲线形态（t10y2y）的动量计算252日Z-Score极值，并叠加动量衰竭二阶导数避免接飞刀，以此作为市场预期的极端Price-in且开始反转的脉冲信号。
    数据: fomc_sentiment, dgs2, t10y2y
    触发: FOMC情绪单日阶梯跳跃突变且Z-Score>1.5；或 dgs2 5日降息动量Z-Score<-2.0且变陡极值且单日下杀动能减弱(二阶衰竭)。看空反之。
    输出: 严格狙击手级脉冲，看多TLT=+1.0，看空=-1.0，常态信号为0.0。
    """

    def __init__(self):
        self.name = 'fomc_policy_pivot_shock_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 常态下强制信号返回 0.0 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['fomc_sentiment', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        df = data[required_cols].ffill()

        # --------------------------------------------------------
        # 1. 边际变化铁律：FOMC文本情绪变化 (处理阶梯数据，将静态转变为脉冲事件)
        # --------------------------------------------------------
        fomc_diff1 = df['fomc_sentiment'].diff(1)
        fomc_diff5 = df['fomc_sentiment'].diff(5)
        
        fomc_diff5_mean = fomc_diff5.rolling(252, min_periods=21).mean()
        fomc_diff5_std = fomc_diff5.rolling(252, min_periods=21).std().replace(0, np.nan)
        fomc_z = (fomc_diff5 - fomc_diff5_mean) / fomc_diff5_std

        # --------------------------------------------------------
        # 2. 边际变化铁律：2年期短端收益率（美联储最敏感指标）预期突变极值
        # --------------------------------------------------------
        dgs2_diff5 = df['dgs2'].diff(5)
        dgs2_mean = dgs2_diff5.rolling(252, min_periods=21).mean()
        dgs2_std = dgs2_diff5.rolling(252, min_periods=21).std().replace(0, np.nan)
        dgs2_z = (dgs2_diff5 - dgs2_mean) / dgs2_std

        # --------------------------------------------------------
        # 3. 边际变化铁律：10年-2年利差（牛陡 Bull Steepening 形态突变）极值
        # --------------------------------------------------------
        t10_2_diff5 = df['t10y2y'].diff(5)
        t10_2_mean = t10_2_diff5.rolling(252, min_periods=21).mean()
        t10_2_std = t10_2_diff5.rolling(252, min_periods=21).std().replace(0, np.nan)
        t10_2_z = (t10_2_diff5 - t10_2_mean) / t10_2_std

        # --------------------------------------------------------
        # 4. 二阶导数铁律：防接飞刀，必须出现动量衰竭确认
        # --------------------------------------------------------
        dgs2_diff1 = df['dgs2'].diff(1)
        dgs2_diff1_avg3 = df['dgs2'].diff(3) / 3.0
        
        # 极端下跌中衰竭：今日的日内下杀幅度弱于前3日均速下杀势头 (动能收敛)
        dgs2_down_exhausted = dgs2_diff1 > dgs2_diff1_avg3
        # 极端上涨中衰竭：今日的日内上涨幅度弱于前3日均速上涨势头 (动能收敛)
        dgs2_up_exhausted = dgs2_diff1 < dgs2_diff1_avg3

        # --------------------------------------------------------
        # 5. 零值休眠铁律：脉冲触发逻辑
        # --------------------------------------------------------
        # 多头触发 (看多美债):
        # A. FOMC突发鸽派大转弯：会议声明文本当日瞬间大幅跃升鸽派
        cond_fomc_long = (fomc_diff1 > 0) & (fomc_z > 1.5)
        # B. 宏观定价过激后衰竭：短端急跌定价骤进降息期且曲线牛陡，叠加下探动能衰减
        cond_macro_long = (dgs2_z < -2.0) & (t10_2_z > 1.5) & dgs2_down_exhausted

        # 空头触发 (看空美债):
        # A. FOMC突发鹰派大转弯：会议声明文本当日瞬间大幅下坠鹰派
        cond_fomc_short = (fomc_diff1 < 0) & (fomc_z < -1.5)
        # B. 宏观定价过激后衰竭：短端急涨定价骤进加息期且曲线熊平，叠加上攻动能衰减
        cond_macro_short = (dgs2_z > 2.0) & (t10_2_z < -1.5) & dgs2_up_exhausted

        signal[cond_fomc_long | cond_macro_long] = 1.0
        signal[cond_fomc_short | cond_macro_short] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"