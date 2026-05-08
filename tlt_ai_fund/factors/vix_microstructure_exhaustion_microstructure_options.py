import numpy as np
import pandas as pd

class VixMicrostructureExhaustionFactor:
    """波动率微观结构衰竭反转因子 (microstructure/options)

    逻辑: 波动率期权(VIX)具有极强的均值回复特性。当VIX因恐慌集中爆发而飙升至极值时，往往伴随跨资产的流动性无差别抛售(包括美债)；而当VIX停止创新高并开始回落时(二阶导衰竭)，标志着流动性危机解除与央行救市预期升温，此时是避险买盘回流、做多美债的绝佳脉冲点。反之，当VIX长期低迷极度自满时突然向上爆发，往往是对冲资金紧急买入期权、紧缩冲击或系统性风险的开始，利空美债。
    数据: vixcls (CBOE VIX 波动率指数)
    触发: 多头脉冲: VIX 252日水位或5日动量 Z-Score > 2.0 (极度恐慌)，且今日 VIX 环比下降并低于3日均值 (恐慌衰竭)；空头脉冲: VIX Z-Score < -1.5 (极度自满)，且单日跳升突破过去20天的日均波动水平 (异动爆发)。
    输出: +1.0 看多美债 (恐慌消退抄底)，-1.0 看空美债 (自满破灭逃顶)，常态 0.0 处于休眠
    """

    def __init__(self):
        self.name = 'vix_microstructure_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 常态下必须保持零值休眠
        signal = pd.Series(0.0, index=data.index)
        
        # 纯粹使用自有领域的高价值衍生数据
        if 'vixcls' not in data.columns:
            signal.name = self.name
            return signal

        # 获取 VIX 数据并处理基础缺失值
        vix = data['vixcls'].ffill()
        
        if vix.dropna().empty:
            signal.name = self.name
            return signal
            
        # 1. 计算 VIX 长期水位的 Z-Score (反映情绪的绝对极值)
        # 使用 252 个交易日代表一年的自然宏观周期
        roll_mean = vix.rolling(window=252, min_periods=126).mean()
        roll_std = vix.rolling(window=252, min_periods=126).std()
        vix_zscore = (vix - roll_mean) / roll_std
        
        # 2. 计算 VIX 微观动量脉冲的 Z-Score (反映恐慌情绪的边际爆发烈度)
        # 使用 5 日差值代表极短期的微观期权买盘脉冲
        vix_diff5 = vix.diff(5)
        diff5_mean = vix_diff5.rolling(window=252, min_periods=126).mean()
        diff5_std = vix_diff5.rolling(window=252, min_periods=126).std()
        vix_mom_zscore = (vix_diff5 - diff5_mean) / diff5_std
        
        # 3. 衰竭与突变条件 (严格遵守二阶导数铁律：禁止只看极值，必须有边际反转)
        
        # 多头条件之衰竭确认：今日 VIX 必须环比下降，且跌破极短期的 3 日移动均线
        vix_drop_exhaustion = (vix.diff(1) < 0) & (vix < vix.rolling(window=3).mean())
        
        # 空头条件之突变确认：单日 VIX 飙升，跳升幅度必须大于过去 20 日(月度)的历史单日波动率，反映突发异动
        daily_diff = vix.diff(1)
        roll_daily_std = daily_diff.rolling(window=20, min_periods=10).std()
        vix_surge_breakout = (daily_diff > 0) & (vix > vix.rolling(window=3).mean()) & (daily_diff > roll_daily_std)
        
        # 4. 组合触发信号 (狙击手级脉冲逻辑)
        
        # 多头触发：前一日处于恐慌极高水位 或 经历过极端的动量飙升，且今日确认衰竭拐点
        panic_extreme = (vix_zscore.shift(1) > 2.0) | (vix_mom_zscore.shift(1) > 2.0)
        long_cond = panic_extreme & vix_drop_exhaustion
        
        # 空头触发：前一日处于极度低迷和自满的状态，且今日突发剧烈反弹(波动率空头踩踏)
        complacency_extreme = (vix_zscore.shift(1) < -1.5)
        short_cond = complacency_extreme & vix_surge_breakout
        
        # 严格赋值，其余不符合条件的维持初始的 0.0 (Sniper 休眠态)
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"