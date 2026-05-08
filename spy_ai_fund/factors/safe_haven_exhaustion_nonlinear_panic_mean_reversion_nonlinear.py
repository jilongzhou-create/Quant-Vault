import numpy as np
import pandas as pd

class YenCarryUnwindExhaustionFactor:
    """Yen Carry Unwind Exhaustion (panic_mean_reversion/nonlinear)

    逻辑: 恐慌抛售往往伴随日元套息交易平仓(去杠杆导致USD/JPY暴跌)。当VIX处于高位但今日跌破5日均线(恐慌二阶导转负，即抛售衰竭)，且USD/JPY近一周停止下跌(去杠杆踩下刹车)时，输出强看多信号(+1.0)抄底；相反，在极端自满(VIX极低)时若VIX突升且日元开始升值，输出看空信号(-1.0)防范流动性紧缩主跌浪。
    数据: vixcls (VIX恐慌指数), dexjpus (美日汇率)
    输出: +1.0 (恐慌衰竭，做多SPY), -1.0 (自满被打破，做空SPY), 0.0 (常态休眠)
    触发条件: 预期Trigger Rate 8%-12%。VIX_Z > 0.8且跌破一周均线且日元停止升值时看多；VIX_Z < -0.5且突破一周均线且日元开始升值时看空。
    """

    def __init__(self):
        self.name = 'yen_carry_unwind_exhaustion_panic_mean_reversion_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 默认返回全0 Series
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必需的交叉数据列是否存在
        if 'vixcls' not in data.columns or 'dexjpus' not in data.columns:
            return signal
            
        # 前向填充缺失值
        vix = data['vixcls'].ffill()
        usdjpy = data['dexjpus'].ffill()

        # 检查数据是否全部为空
        if vix.isna().all() or usdjpy.isna().all():
            return signal

        # 经济学物理阈值：
        # 252: 一年交易日，用于锚定宏观波动率的Z-Score基准
        # 5: 一周交易日，用于判断短期恐慌/去杠杆的脉冲转折
        vix_mean_1y = vix.rolling(window=252, min_periods=126).mean()
        vix_std_1y = vix.rolling(window=252, min_periods=126).std()
        
        # 1. 恐慌极端度计算 (捕捉极值)
        vix_z = (vix - vix_mean_1y) / vix_std_1y
        
        # 2. 衰竭/爆发转折计算 (防接飞刀，二阶导数铁律)
        vix_5sma = vix.rolling(window=5).mean()
        
        # 3. 宏观流动性确认 - 美日汇率的5日动量 (判断日元套息交易是否正在平仓)
        usdjpy_momentum = usdjpy.diff(5)

        # 【做多逻辑】 (极度恐慌 + 抛售衰竭 + 宏观去杠杆结束)
        # 1. 恐慌极值: vix_z > 0.8 (宏观高危区间，此时盲目接飞刀易死)
        # 2. 抛售衰竭: vix < vix_5sma (今日VIX明确跌破短期均线，恐慌势头彻底逆转)
        # 3. 流动性确认: usdjpy_momentum > 0.0 (日元停止升值，避险套息资金去杠杆暂缓，确认底部的坚实度)
        buy_cond = (vix_z > 0.8) & (vix < vix_5sma) & (usdjpy_momentum > 0.0)

        # 【做空逻辑】 (极度自满 + 突发冲击 + 宏观去杠杆开始)
        # 1. 极度自满: vix_z < -0.5 (美股长牛背景下的绝对低波动安全区)
        # 2. 突发冲击: vix > vix_5sma (今日VIX跳升突破短期均线，警报拉响)
        # 3. 流动性确认: usdjpy_momentum < 0.0 (日元开始升值，套息资金实质性开启撤退避险，SPY将失去支撑)
        sell_cond = (vix_z < -0.5) & (vix > vix_5sma) & (usdjpy_momentum < 0.0)

        # 信号赋值
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0

        signal.name = self.name
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}()"