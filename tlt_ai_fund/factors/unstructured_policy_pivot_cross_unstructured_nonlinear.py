import numpy as np
import pandas as pd

class UnstructuredPolicyPivotCrossFactor:
    """Policy Pivot Shock Cross Factor (unstructured/nonlinear)

    逻辑: 捕捉美联储政策预期突变的极端脉冲。将FOMC情绪得分的阶梯性突变、2年期短端利率(dgs2)的断崖下行、以及收益率曲线(t10y2y)的急剧变陡(Bull Steepening)进行高维交叉。这是一个典型的“狙击手”脉冲因子，绝非常态连续因子。
    数据: fomc_sentiment, dgs2, t10y2y
    触发: 组合 Pivot Score 的短期 Z-Score > 2.5 (极端突变发生) AND 组合得分下穿3日均线 (动能极值已过，衰竭反转，防接飞刀)
    输出: [-1.0, 1.0] 的脉冲信号，+1.0=极度鸽派突变且衰竭，-1.0=极度鹰派突变且衰竭。其余时间静默(0.0)。
    """

    def __init__(self):
        self.name = 'unstructured_policy_pivot_cross'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始化全0
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['fomc_sentiment', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal

        # 填充低频和节假日缺失值
        df = data[required_cols].ffill()

        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝对禁止直接使用fomc_sentiment或利率水平值，必须计算5日窗口的变化量，捕捉“预期突发变化”的瞬间
        fomc_chg = df['fomc_sentiment'].diff(5)
        dgs2_chg = df['dgs2'].diff(5)
        curve_chg = df['t10y2y'].diff(5)

        # 对各边际变化维度计算长期标准分数(252日)，使其具备统一的横向可比性
        fomc_z = (fomc_chg - fomc_chg.rolling(252).mean()) / fomc_chg.rolling(252).std()
        dgs2_z = (dgs2_chg - dgs2_chg.rolling(252).mean()) / dgs2_chg.rolling(252).std()
        curve_z = (curve_chg - curve_chg.rolling(252).mean()) / curve_chg.rolling(252).std()

        # 方法C: 非线性特征交叉
        # 构造高维政策转向综合得分(Pivot Score): 
        # 鸽派转向 = 情绪边际变鸽(+) + 短端利率边际暴跌(取负为+) + 曲线急剧陡峭(+)
        pivot_score = fomc_z.fillna(0.0) - dgs2_z.fillna(0.0) + curve_z.fillna(0.0)

        # 提取组合得分的短期爆发脉冲 (63日窗口捕捉近期突发极端波动)
        pulse_z = (pivot_score - pivot_score.rolling(63).mean()) / pivot_score.rolling(63).std()

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 绝对禁止在单边主跌浪中接飞刀，必须等待突发脉冲动能跨越极值点，并向3日均值回归（衰竭）
        exhaustion_long = pivot_score < pivot_score.rolling(3).mean()
        exhaustion_short = pivot_score > pivot_score.rolling(3).mean()

        # 脉冲信号生成: 极端异动 + 动能衰竭 = 狙击点
        long_cond = (pulse_z > 2.5) & exhaustion_long
        short_cond = (pulse_z < -2.5) & exhaustion_short

        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"