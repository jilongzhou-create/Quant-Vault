import numpy as np
import pandas as pd

class GoldVolPanicExhaustionFactor:
    """黄金波动率极值与恐慌衰竭因子 (microstructure)

    逻辑: 黄金期权隐含波动率(GVZ)是衡量市场极端流动性挤兑与通胀/地缘恐慌的核心微观结构指标。在流动性危机的恐慌极点，美债常常因抛售换现而遭遇主跌浪（严禁直接接飞刀）。只有当 GVZ 升至一年期极端高位（Z-Score > 1.5），且微观动量确认见顶回落（当日明显下降且跌破3日均线）时，说明恐慌开始实质性衰竭，避险资金重返美债，此时触发极短期的高胜率看多脉冲。
    数据: gvzcls (黄金波动率指数)
    触发: 极值(252日 Z-Score > 1.5) + 衰竭二阶导(diff < 0 且跌破3日均线)。反转看空同理。
    输出: +1.0 看多TLT脉冲; -1.0 看空脉冲。常态时保持零值休眠 (Sniper Pulse)，信号触发后维持3日。
    """

    def __init__(self):
        self.name = 'gold_vol_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (常态信号为 0.0)
        signal = pd.Series(0.0, index=data.index)
        
        # 处理缺失必要数据列的情况
        if 'gvzcls' not in data.columns:
            return signal
            
        # 获取黄金隐含波动率数据并前向填充
        gvz = data['gvzcls'].ffill()
        
        # 1. 宏观极端水位锚定 (252交易日代表1个自然年的宏观资金记忆基准)
        roll_mean_252 = gvz.rolling(window=252, min_periods=126).mean()
        roll_std_252 = gvz.rolling(window=252, min_periods=126).std()
        
        # 防护除零错误
        roll_std_252 = roll_std_252.replace(0, np.nan)
        gvz_zscore = (gvz - roll_mean_252) / roll_std_252
        
        # 2. 微观结构动量与边际变化 (3日代表期权微观市场的 T+3 短期结算均值)
        ma_3 = gvz.rolling(window=3, min_periods=2).mean()
        
        # 铁律3: 边际变化 (捕获预期的边际突变)
        gvz_diff = gvz.diff(1)
        
        # 3. 铁律2: 二阶导数 (极值 + 衰竭) 严防接飞刀!
        # 看多美债脉冲: 恐慌指标升至历史极端尾部(Z>1.5)，且在边际上确认回落(diff<0 且破3日均线)
        trigger_long = (gvz_zscore > 1.5) & (gvz_diff < 0) & (gvz < ma_3)
        
        # 看空美债脉冲: 平静自满达到极点(Z<-1.5)，且边际上确认风险重估抬头(diff>0 且上穿3日均线)
        trigger_short = (gvz_zscore < -1.5) & (gvz_diff > 0) & (gvz > ma_3)
        
        # 4. 极短期脉冲维持
        # 确保满足目标 Trigger Rate (5% - 15%)，将触发当日及随后连续2日(共3日)视为脉冲窗口
        pulse_long = trigger_long.rolling(window=3, min_periods=1).max() > 0
        pulse_short = trigger_short.rolling(window=3, min_periods=1).max() > 0
        
        # 赋值并排斥冲突信号
        signal.loc[pulse_long] = 1.0
        signal.loc[pulse_short & ~pulse_long] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"