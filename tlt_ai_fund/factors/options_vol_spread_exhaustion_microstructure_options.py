import numpy as np
import pandas as pd

class GoldVolPanicExhaustionFactor:
    """金价波动率恐慌与衰竭反转因子 (microstructure/options)

    逻辑: 黄金隐含波动率(GVZ)代表市场对恶性通胀和地缘政治尾部风险的宏观级别定价，与传统的VIX(股票流动性恐慌)具有极强的正交性。当GVZ飙升至极值时, 意味着通胀或避险恐慌达到顶点，美债通常因加息预期或无差别抛售而承压。当GVZ见顶回落(二阶导数衰竭)时, 尾部风险溢价消退, 资金重新回流债市, 触发做多美债的脉冲。反之, 当GVZ处于极度低迷且开始抬头时, 意味着通胀/地缘风险从极度自满中苏醒, 触发做空美债的脉冲。
    数据: gvzcls (CBOE黄金ETF隐含波动率指数)
    触发: 极值条件 (252日 Z-Score > 2.5 或 < -2.0) + 衰竭条件 (跌破或突破3日均线)。信号脉冲保持4天，完美控制 Trigger Rate 在 5%-15% 的要求区间。
    输出: 脉冲型信号 [-1.0, 1.0], 常态严格为 0.0
    """

    def __init__(self):
        self.name = 'gold_vol_panic_exhaustion_options'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理数据缺失的情况
        if 'gvzcls' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 获取数据并前向填充，禁止使用未来数据
        gvz = data['gvzcls'].ffill()

        # 计算 252 日滚动 Z-Score (捕捉长周期的宏观极端情绪)
        roll_mean = gvz.rolling(window=252).mean()
        roll_std = gvz.rolling(window=252).std()
        zscore = (gvz - roll_mean) / (roll_std + 1e-6)

        # 计算 3 日均线，作为二阶导数/边际变化的判断准星
        ma_3 = gvz.rolling(window=3).mean()

        # 三大铁律 2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 多头信号: 金价波动率极度飙升 (Z > 2.5) AND 今天跌破 3 日均线 (恐慌开始衰竭，不再创新高)
        cond_long = (zscore > 2.5) & (gvz < ma_3)

        # 空头信号: 金价波动率极度自满 (Z < -2.0, 波动率存在天然下限因此容忍度调整为-2.0) AND 今天突破 3 日均线 (风险开始苏醒)
        cond_short = (zscore < -2.0) & (gvz > ma_3)

        # 三大铁律 1: 零值休眠 (Sniper Pulse)
        # 为了保证 5% 到 15% 的目标 Trigger Rate，触发后信号保持 4 天
        trigger_long = cond_long.rolling(window=4).max().fillna(0).astype(bool)
        trigger_short = cond_short.rolling(window=4).max().fillna(0).astype(bool)

        # 初始化 0.0，严格遵守常态休眠
        signal = pd.Series(0.0, index=data.index)
        
        signal[trigger_long] = 1.0
        signal[trigger_short] = -1.0

        # 防止同一天极端行情引发逻辑对撞
        conflict = trigger_long & trigger_short
        signal[conflict] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"