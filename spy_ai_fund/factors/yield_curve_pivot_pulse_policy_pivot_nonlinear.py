import numpy as np
import pandas as pd

class YieldCurvePivotPulseFactor:
    """政策预期突变冲量因子 (policy_pivot/nonlinear)

    逻辑: 通过短端利率(2年期)的极短期动量突变结合收益率曲线变陡/变平，捕捉美联储政策转向(鹰/鸽)的瞬时定价窗口。
    数据: dgs2 (2年期美债), t10y2y (10年-2年期限利差)
    输出: 市场抢跑降息且曲线牛陡时输出+1.0(看多)，加息预期突增且曲线熊平时输出-1.0(看空)。
    触发条件: DGS2的5日变化剧烈(>15bp)，同时带有强烈的期限利差配合。预期Trigger Rate 8-12%。
    """

    def __init__(self):
        self.name = 'yield_curve_pivot_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 默认返回全0的脉冲信号
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖列是否存在
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return signal

        # 数据前向填充，防止个别节假日数据缺失导致差分断层
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # 计算边际变化动量：5个交易日(一周)内的预期重定价冲量
        # 15个基点(0.15%)反映了超过半次标准降息/加息的预期突变，代表宏观资金的急速抢跑
        dgs2_5d_diff = dgs2.diff(5)
        # 5个基点(0.05%)反映了曲线结构的同步配合变化(确认是受短端预期驱动)
        t10y2y_5d_diff = t10y2y.diff(5)

        # 宏观微观趋势状态确认(季度均线)，确保属于动能释放而不是震荡噪音
        dgs2_quarter_mean = dgs2.rolling(window=63, min_periods=21).mean()
        
        # 鸽派突发转向(Dovish Pivot) -> 强看多 (+1.0)
        # 物理意义: 2年期利率短时间内剧烈下行，同时长短期利差变陡(短端下行更狠，典型的Bull Steepening)，且本身已处于政策转松趋势
        is_dovish_shock = (dgs2_5d_diff <= -0.15)
        is_bull_steepening = (t10y2y_5d_diff >= 0.05)
        dovish_trend = (dgs2 < dgs2_quarter_mean)
        long_cond = is_dovish_shock & is_bull_steepening & dovish_trend

        # 鹰派紧缩恐慌(Hawkish Pivot) -> 看空 (-1.0)
        # 物理意义: 2年期利率短时间内剧烈上行，同时长短期利差迅速走平/倒挂加深(Bear Flattening)，且处于收紧趋势
        is_hawkish_shock = (dgs2_5d_diff >= 0.15)
        is_bear_flattening = (t10y2y_5d_diff <= -0.05)
        hawkish_trend = (dgs2 > dgs2_quarter_mean)
        short_cond = is_hawkish_shock & is_bear_flattening & hawkish_trend

        # 赋值信号 (使用 fillna(False) 防御 NaN 报错)
        signal.loc[long_cond.fillna(False)] = 1.0
        signal.loc[short_cond.fillna(False)] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"