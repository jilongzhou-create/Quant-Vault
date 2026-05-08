import numpy as np
import pandas as pd

class PolicyPivotYieldCurveShockFactor:
    """政策预期突变与曲线陡峭脉冲因子 (Policy Pivot Shock)

    逻辑: 捕捉美联储政策预期的极端突变(此逻辑为血泪教训总结的正确示例)。
          当短端利率(dgs2)暴跌且曲线(t10y2y)急剧变陡时(Bull Steepening)，代表降息预期突然爆发，
          此时看多美债(TLT)。反之，短端急剧飙升且曲线变平(Bear Flattening)代表加息恐慌，看空美债。
          因子严格遵守脉冲特性，仅在极端状态(Z-Score>2.5)且动量开始衰竭(二阶导数反转)时触发狙击手信号。
    数据: dgs2, t10y2y
    触发: dgs2 5日变动 Z-Score < -2.5 AND t10y2y 5日变动 Z-Score > 2.5 AND 短端下跌动量衰竭 (触发+1.0)
    输出: +1.0 或 -1.0 脉冲信号，非触发日保持 0.0 休眠
    """

    def __init__(self):
        self.name = 'policy_pivot_yield_curve_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 必须处理数据缺失情况，防止系统报错
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 避免前瞻偏差，使用前向填充
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # 铁律1: 零值休眠 (常态下必须返回 0.0)
        signal = pd.Series(0.0, index=data.index)

        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝对禁止直接使用绝对水位，计算 5 日变化量捕捉预期突变的瞬间
        dgs2_diff5 = dgs2.diff(5)
        t10y2y_diff5 = t10y2y.diff(5)

        # 计算 252 日(一年)滚动 Z-Score 识别极端脉冲事件
        # 设定 min_periods=60 保证初期有足够数据点计算标准差
        dgs2_mean = dgs2_diff5.rolling(window=252, min_periods=60).mean()
        dgs2_std = dgs2_diff5.rolling(window=252, min_periods=60).std()
        dgs2_z = (dgs2_diff5 - dgs2_mean) / (dgs2_std + 1e-6)

        t10y2y_mean = t10y2y_diff5.rolling(window=252, min_periods=60).mean()
        t10y2y_std = t10y2y_diff5.rolling(window=252, min_periods=60).std()
        t10y2y_z = (t10y2y_diff5 - t10y2y_mean) / (t10y2y_std + 1e-6)

        # 铁律2: 二阶导数衰竭 (Anti-Catch-Falling-Knife)
        # 通过一阶导数与短期均值的比较，捕捉动量衰竭点（二阶导数反转），禁止接飞刀
        dgs2_diff1 = dgs2.diff(1)
        dgs2_diff1_ma3 = dgs2_diff1.rolling(window=3).mean()

        # 暴跌衰竭: dgs2 的单日下跌幅度开始缩小，即当前一阶导大于过去3日平均 (跌势放缓)
        exhaustion_bull = dgs2_diff1 > dgs2_diff1_ma3
        
        # 暴涨衰竭: dgs2 的单日上涨幅度开始缩小，即当前一阶导小于过去3日平均 (涨势放缓)
        exhaustion_bear = dgs2_diff1 < dgs2_diff1_ma3

        # 条件组合 (所有阈值均具经济学意义: Z>2.5代表尾部极端事件)
        
        # 看多脉冲: 短端极度下行 (降息预期) + 曲线极度变陡 + 跌势开始衰竭
        long_cond = (dgs2_z < -2.5) & (t10y2y_z > 2.5) & exhaustion_bull

        # 看空脉冲: 短端极度飙升 (加息恐慌) + 曲线极度变平/倒挂 + 涨势开始衰竭
        short_cond = (dgs2_z > 2.5) & (t10y2y_z < -2.5) & exhaustion_bear

        # 触发狙击手脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"