import numpy as np
import pandas as pd

class GoldOptionsVolShockFactor:
    """黄金期权波动率突变衰竭因子 (unstructured/options)

    逻辑: GVZCLS (CBOE黄金ETF隐含波动率) 刻画了纯粹的宏观避险与流动性恐慌情绪。相比VIX的股市属性，GVZCLS对美联储流动性与宏观货币预期更为敏感。当其发生极其剧烈的边际跳跃（非线性突变）时，意味着流动性危机或宏观恐慌达到极点。配合二阶导数衰竭（高位回落），捕捉恐慌充分Price-in后、避险资金回流长端美债(TLT)的绝佳狙击点。
    数据: gvzcls (CBOE黄金ETF隐含波动率指数)
    触发: GVZCLS 5日边际变化的 252日 Z-Score > 2.5 且动能衰竭（当日回落且低于3日均值） -> 看多美债(+1.0)；Z-Score < -2.5 且触底反弹 -> 看空美债(-1.0)。
    输出: 严格的脉冲型信号，常态休眠返回0.0。
    """

    def __init__(self):
        self.name = 'gold_options_vol_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'gvzcls' not in data.columns:
            return signal
            
        # 获取可用数据并前向填充，处理节假日缺失值
        gvz = data['gvzcls'].ffill()
        
        # 铁律3: 边际变化 (使用 5 日动量，捕捉期权市场宏观预期的剧烈跳跃)
        gvz_mom = gvz.diff(5)
        
        # 计算 252 日 (约1个交易年) 的滚动 Z-Score 
        roll_mean = gvz_mom.rolling(window=252, min_periods=126).mean()
        roll_std = gvz_mom.rolling(window=252, min_periods=126).std()
        
        # 防止除以0导致无穷大
        roll_std = roll_std.replace(0, np.nan)
        z_score = (gvz_mom - roll_mean) / roll_std
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 衰竭确认: 当日势头必须转向，并且低于/高于近期均值以确认拐点
        is_falling = (gvz.diff(1) < 0) & (gvz < gvz.rolling(3).mean())
        is_rising = (gvz.diff(1) > 0) & (gvz > gvz.rolling(3).mean())
        
        # 铁律1: 零值休眠 (Sniper Pulse) 极端极值触发 + 衰竭
        # 多头脉冲: 恐慌挤兑极值 + 开始消退 -> 避险资金买入美债
        bull_signal = (z_score > 2.5) & is_falling
        
        # 空头脉冲: 极度乐观导致避险崩溃 + 情绪触底反弹 -> 抛售美债/利率预期上调
        bear_signal = (z_score < -2.5) & is_rising
        
        signal.loc[bull_signal] = 1.0
        signal.loc[bear_signal] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"