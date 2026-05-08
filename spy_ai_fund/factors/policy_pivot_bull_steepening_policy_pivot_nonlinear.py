import numpy as np
import pandas as pd

class PolicyPivotBullSteepeningFactor:
    """政策转向与牛陡脉冲因子 (policy_pivot/nonlinear)

    逻辑: 捕捉短端利率剧烈下行导致收益率曲线突然'变陡'(Bull Steepening)的瞬间。当2年期美债收益率在1周内急跌，且10Y-2Y利差急速走阔时，代表市场在强烈抢跑美联储降息。此时若高收益债信用利差未见飙升(即排除经济硬着陆和衰退恐慌)，则说明这是纯粹的流动性宽松预期突变，产生强烈看多美股的脉冲。反之，短端暴涨且曲线变平则为紧缩恐慌，触发看空脉冲。
    数据: [dgs2, t10y2y, bamlh0a0hym2]
    输出: +1.0 看多(纯流动性宽松突变), -1.0 看空(紧缩预期骤起)
    触发条件: 2年期美债收益率5日下跌>15bp 且 利差走阔>10bp 且 信用利差变化<=5bp 触发+1.0。脉冲化处理，预期 Trigger Rate 约 5%-12%。
    """

    def __init__(self, lookback_window: int = 5, rate_drop_bps: float = 0.15, curve_steep_bps: float = 0.10, credit_filter_bps: float = 0.05):
        self.name = 'policy_pivot_bull_steepening'
        # 5日(一周)作为捕捉动量突变的窗口
        self.lookback_window = lookback_window
        # 15个基点代表短期利率预期的剧烈变化(超过单次25bp加息/降息预期的一半)
        self.rate_drop_bps = rate_drop_bps
        # 10个基点的期限利差变化代表曲线形状发生实质性扭曲
        self.curve_steep_bps = curve_steep_bps
        # 信用利差容忍度，用于过滤因衰退危机造成的被动陡峭
        self.credit_filter_bps = credit_filter_bps

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        req_cols = ['dgs2', 't10y2y', 'bamlh0a0hym2']
        if not all(col in data.columns for col in req_cols):
            return signal

        # 债券市场数据可能存在节假日缺失，向前填充对齐
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()

        # 计算短期边际变化 (二阶导数思想: 关注预期的动量，而不是绝对水位)
        dgs2_chg = dgs2.diff(self.lookback_window)
        t10y2y_chg = t10y2y.diff(self.lookback_window)
        hy_spread_chg = hy_spread.diff(self.lookback_window)

        # Bull Steepening: 短端利率骤降 + 曲线变陡 + 信用利差未见极端恶化 (排除危机) -> 流动性狂欢 (多头脉冲)
        bull_steepening = (dgs2_chg < -self.rate_drop_bps) & \
                          (t10y2y_chg > self.curve_steep_bps) & \
                          (hy_spread_chg <= self.credit_filter_bps)

        # Bear Flattening: 短端利率骤升 + 曲线变平 -> 紧缩超预期 (空头脉冲)
        bear_flattening = (dgs2_chg > self.rate_drop_bps) & \
                          (t10y2y_chg < -self.curve_steep_bps)

        # 转换为"狙击手"级别的极窄脉冲: 只在预期状态跃变的瞬间(第一天)触发信号
        buy_pulse = bull_steepening.astype(int).diff() == 1
        sell_pulse = bear_flattening.astype(int).diff() == 1

        signal.loc[buy_pulse] = 1.0
        signal.loc[sell_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback={self.lookback_window}d, rate_thr={self.rate_drop_bps}, curve_thr={self.curve_steep_bps})"