import numpy as np
import pandas as pd

class UnstructuredRealRateVolReversalFactor:
    """Unstructured Real Rate Volatility Reversal (volatility/unstructured)

    逻辑: 结合基于NLP的新闻经济政策不确定性(EPU)与黄金波动率(实际利率跨资产恐慌代理)，合成宏观恐慌指数。
          脉冲信号要求恐慌极值出现且开始衰竭。为避免接飞刀并提高CondIC，通过FOMC文本情绪动态切换方向：
          1. 鹰派周期(FOMC情绪<0)：市场处于加息/通胀恐慌，美债超卖。恐慌衰竭意味着利空出尽，做多美债(+1.0)。
          2. 鸽派周期(FOMC情绪>=0)：市场处于衰退/避险恐慌，美债超买。恐慌衰竭意味着避险情绪消退，做空美债(-1.0)。
    数据: usepuindxd (EPU), gvzcls (Gold Vol), fomc_sentiment (FOMC情绪)
    触发: 综合Z-Score > 2.0 且 连续回落衰竭
    输出: 脉冲型, +1.0 或 -1.0
    """

    def __init__(self):
        self.name = 'unstructured_real_rate_vol_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需字段
        required_cols = ['usepuindxd', 'gvzcls', 'fomc_sentiment']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return signal
            
        df = data[required_cols].ffill()
        
        # 1. 经济政策不确定性 (EPU) - NLP非结构化数据
        # 使用 log1p 平滑极端异常值
        epu = np.log1p(df['usepuindxd'])
        epu_mean = epu.rolling(window=126, min_periods=21).mean()
        epu_std = epu.rolling(window=126, min_periods=21).std()
        epu_z = (epu - epu_mean) / (epu_std + 1e-6)
        
        # 2. 黄金波动率 (GVZ) - 实际利率恐慌代理
        gvz = df['gvzcls']
        gvz_mean = gvz.rolling(window=126, min_periods=21).mean()
        gvz_std = gvz.rolling(window=126, min_periods=21).std()
        gvz_z = (gvz - gvz_mean) / (gvz_std + 1e-6)
        
        # 3. 合成宏观恐慌脉冲指数
        # 避免某单一资产的噪音，要求共振
        combo_vol = epu_z + gvz_z
        
        # 4. 定义极值与衰竭条件 (铁律2: 二阶导数)
        # 极值: 综合 Z-Score > 2.0 (进入恐慌区间)
        extreme_condition = combo_vol > 2.0
        
        # 衰竭: 开始回落，低于3日均值且当日边际变弱
        exhaustion_condition = (combo_vol < combo_vol.rolling(window=3).mean()) & (combo_vol.diff() < 0)
        
        # 5. 确定宏观周期方向 (铁律3: 边际变化与动态映射)
        # 取过去半年(126天)的平均FOMC情绪作为中期货币政策状态
        fomc_regime = df['fomc_sentiment'].rolling(window=126, min_periods=21).mean()
        
        # 6. 生成信号
        trigger = extreme_condition & exhaustion_condition
        
        # 鹰派周期下恐慌衰竭 -> 加息计价完成，资金停止抛售美债 -> 看多 (+1.0)
        signal.loc[trigger & (fomc_regime < 0)] = 1.0
        
        # 鸽派周期下恐慌衰竭 -> 避险结束，资金流出美债 -> 看空 (-1.0)
        signal.loc[trigger & (fomc_regime >= 0)] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"