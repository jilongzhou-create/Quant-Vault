import numpy as np
import pandas as pd

class EpuNewsSurgeExhaustionFactor:
    """新闻恐慌极值衰竭因子 (panic_mean_reversion / unstructured)

    逻辑: 每日经济政策不确定性指数(Daily News EPU)基于海量新闻报纸的NLP分析评估宏观不确定性。美股具有极强的均值回归物理属性：
          1. 新闻恐慌见顶衰竭：不确定性飙升至极值区域后，一旦单日出现显著回落，即为"靴子落地"，由于恐慌情绪被过度消化，此时触发抄底买点。
          2. 突发性恐慌恶化：若在风平浪静的中低位期，不确定性突发大幅飙升并突破近期新高，则是主跌浪恶化的前兆，触发趋势看空避险信号。
    数据: usepuindxd (Daily News EPU)
    输出: 强烈看多美股为 +1.0，恐慌抬头看空为 -1.0，常态休眠返回 0.0
    触发条件: Z-score > 1.2 且动量快速跌破3日均值触发多头；Z-score <= 0.5 且动量飙升破10日新高触发空头，预期 Trigger Rate 约 8%-12%
    """

    def __init__(self):
        self.name = 'epu_news_surge_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律：处理缺少必须列的情况
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 1. 基础数据提取与前向填充 (避免节假日带来的 NaN 丢失)
        epu = data['usepuindxd'].ffill()
        
        # 新闻不确定性指数具有典型的高偏度和厚尾性，取自然对数以稳定方差，避免因个别极端值导致统计失效
        log_epu = np.log1p(epu)
        
        # 2. 设定 126 日 (半年交易日) 为宏观基准窗口，计算滚动 Z-Score
        roll_mean = log_epu.rolling(window=126, min_periods=21).mean()
        roll_std = log_epu.rolling(window=126, min_periods=21).std()
        
        # 避免除以0导致的 inf，极早期数据用后向填充
        roll_std = roll_std.replace(0, np.nan).ffill().bfill()
        
        epu_z = (log_epu - roll_mean) / roll_std
        z_diff = epu_z.diff()
        
        # 计算短线基准：3日均值(短期中枢) 和 10日最高点(近期压力位)
        epu_ma3 = log_epu.rolling(window=3, min_periods=1).mean()
        epu_max10 = log_epu.rolling(window=10, min_periods=5).max()
        
        # 3. 严格二阶导数识别与脉冲休眠初始化
        signal = pd.Series(0.0, index=data.index)
        
        # 抄底信号 (+1.0): 极值 + 衰竭 (防接飞刀)
        # 条件：昨日不确定性处于前 11% 的极高危区域(Z>=1.2)
        #       今日 Z-Score 出现单日强力回落(<= -0.4)，且绝对值直接跌破过去3日的短期中枢
        buy_cond = (
            (epu_z.shift(1) >= 1.2) & 
            (z_diff <= -0.4) & 
            (log_epu < epu_ma3.shift(1))
        )
        
        # 看空信号 (-1.0): 常态平静 + 突发暴雷
        # 条件：昨日处于风平浪静的中低状态(Z<=0.5)
        #       今日突发利空大爆发(Z飙升 >= 0.8)，且强度一举突破过去两周(10日)以来的最高点
        sell_cond = (
            (epu_z.shift(1) <= 0.5) & 
            (z_diff >= 0.8) & 
            (log_epu > epu_max10.shift(1))
        )
        
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        signal.name = self.name
        
        # 确保无多余 NaN 进入下游
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}()"