import numpy as np
import pandas as pd

class VixCurveSteepeningPulseFactor:
    """波动率与期限结构变陡共振反转因子 (volatility/options)

    逻辑: 恐慌抛售导致VIX狂飙时往往伴随美债流动性枯竭被无差别抛售(如2020年3月或2022年)，此时直接买入美债犹如接飞刀。真正的做多脉冲必须等待恐慌极值开始衰竭(VIX回落)，且同时收益率曲线短端急剧下行导致曲线突然变陡(Bull Steepening，市场快速定价美联储降息救市预期)，这种跨资产的共振脉冲才是抄底长债的绝佳时机。反之常态下进入零值休眠。
    数据: vixcls (CBOE隐含波动率), t10y2y (10年-2年国债利差)
    触发: 多头: VIX 252日Z-Score > 2.5 且开始回落 + t10y2y 5日边际变动 Z-Score > 1.5; 空头: VIX Z-Score < -2.0 且开始回升 + t10y2y 5日边际变动 Z-Score < -1.5
    输出: 严格脉冲信号。满足多头条件脉冲 +1.0，满足空头条件脉冲 -1.0，其余时间常态 0.0
    """

    def __init__(self, vix_z_long=2.5, vix_z_short=-2.0, spread_z_thresh=1.5, window=252):
        self.name = 'vix_curve_steepening_pulse'
        self.vix_z_long = vix_z_long
        self.vix_z_short = vix_z_short
        self.spread_z_thresh = spread_z_thresh
        self.window = window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 - 初始常态绝对为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 向前填充处理非交易日及小幅数据缺失
        vix = data['vixcls'].ffill()
        spread = data['t10y2y'].ffill()
        
        # 1. 波动率极值水平 (252个交易日约一年)
        vix_mean = vix.rolling(window=self.window).mean()
        vix_std = vix.rolling(window=self.window).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        # 铁律2: 二阶导数 - 必须等恐慌/贪婪极值出现衰竭才触发，防接飞刀
        vix_rolling_3 = vix.rolling(window=3).mean()
        vix_falling = vix < vix_rolling_3
        vix_rising = vix > vix_rolling_3
        
        # 铁律3: 边际变化 - 绝对不使用利差水平本身(不管是否倒挂)，只看曲线动能(短期急剧变陡或走平)
        spread_diff = spread.diff(5)
        spread_diff_mean = spread_diff.rolling(window=self.window).mean()
        spread_diff_std = spread_diff.rolling(window=self.window).std()
        spread_diff_z = (spread_diff - spread_diff_mean) / spread_diff_std.replace(0, np.nan)
        
        # 触发条件评估
        # 多头脉冲：恐慌极值(Z>2.5) + 开始消退(回落) + 曲线突然极速变陡(Bull Steepening确认降息预期)
        long_cond = (vix_z > self.vix_z_long) & vix_falling & (spread_diff_z > self.spread_z_thresh)
        
        # 空头脉冲：极度贪婪(Z<-2.0) + 开始反抽(回升) + 曲线突然极速走平/倒挂加深(Bear Flattening确认紧缩加息预期)
        short_cond = (vix_z < self.vix_z_short) & vix_rising & (spread_diff_z < -self.spread_z_thresh)
        
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, vix_long={self.vix_z_long}, spread_z={self.spread_z_thresh})"