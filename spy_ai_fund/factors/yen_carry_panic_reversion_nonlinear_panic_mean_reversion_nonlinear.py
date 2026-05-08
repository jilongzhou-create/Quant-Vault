import numpy as np
import pandas as pd

class SafehavenPanicExhaustionPulseFactor:
    """安全避险恐慌极值与衰竭因子 (panic_mean_reversion/nonlinear)

    逻辑: 引入全新的经济学维度——跨资产避险情绪。结合广义美元指数(全球流动性枯竭/现金为王)与黄金波动率(终极地缘与宏观不确定性)。当二者组成的综合宏观恐慌指数达到极端高位后开始回落，标志着全球避险情绪衰竭，美股迎来高胜率抄底买点；而在恐慌指数缓慢发酵上升期，输出看空信号防范钝刀割肉。完全规避了传统VIX与信用利差因子的拥挤重合。
    数据: dtwexbgs(广义美元指数), gvzcls(黄金VIX)
    输出: 强看多(+1.0)表示避险恐慌极值见顶回落，看空(-1.0)表示避险情绪正在温水煮青蛙般恶化
    触发条件: 综合恐慌Z-Score > 1.5且今日回落触发+1.0，0 < Z-Score <= 1.5且双双上涨触发-1.0。预期 Trigger Rate 10%-15%
    """

    def __init__(self):
        self.name = 'safehaven_panic_exhaustion_pulse_panic_mean_reversion_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必要数据字段是否存在
        if 'dtwexbgs' not in data.columns or 'gvzcls' not in data.columns:
            return signal

        # 缺失值前向填充
        usd = data['dtwexbgs'].ffill()
        gvz = data['gvzcls'].ffill()

        # 计算252日滚动Z-Score (无未来函数)
        usd_mean = usd.rolling(window=252, min_periods=63).mean()
        usd_std = usd.rolling(window=252, min_periods=63).std()
        usd_z = (usd - usd_mean) / (usd_std + 1e-6)

        gvz_mean = gvz.rolling(window=252, min_periods=63).mean()
        gvz_std = gvz.rolling(window=252, min_periods=63).std()
        gvz_z = (gvz - gvz_mean) / (gvz_std + 1e-6)

        # 综合全球宏观避险恐慌指数
        panic_index = usd_z + gvz_z

        # 边际动量变化
        usd_diff = usd.diff(1)
        gvz_diff = gvz.diff(1)
        panic_diff = panic_index.diff(1)

        # 1. 强看多 (抄底): 恐慌极值 + 衰竭
        # 昨天处于极端恐慌 (综合Z-Score > 1.5), 今天恐慌指数整体回落，且黄金恐慌(GVZ)明确下降
        long_cond = (panic_index.shift(1) > 1.5) & (panic_diff < 0) & (gvz_diff < 0)

        # 2. 看空 (防接飞刀): 钝刀割肉期
        # 恐慌指数在历史均值之上(> 0.0)但未到极值(<= 1.5)，且美元和黄金恐慌今天双双上升
        short_cond = (panic_index > 0.0) & (panic_index <= 1.5) & (usd_diff > 0) & (gvz_diff > 0)

        # 写入信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        # 处理可能出现的NaN
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"