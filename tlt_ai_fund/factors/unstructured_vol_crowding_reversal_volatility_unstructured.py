import numpy as np
import pandas as pd

class UnstructuredVolCrowdingReversalFactor:
    """非结构化跨资产波动率衰竭因子 (波动率极值与拥挤反转 / NLP情绪)

    逻辑: 结合跨资产波动率极值与NLP文本情绪，捕捉极端宏观对冲拥挤的反转。常态下处于零值休眠。
          当市场极度恐慌(VIX极值飙升)且避险资产波动率(GVZCLS)高企时，往往形成看空美债的拥挤盘。
          只有当波动率指标开始同步衰竭(二阶导数<0)，且此时美联储传递出边际鸽派的文本突变(NLP情感陡升)时，
          才标志着极度拥挤盘开始瓦解，此时触发狙击手级的做多美债脉冲。反之亦然。
    数据: vixcls (VIX指数), gvzcls (黄金波动率), fomc_sentiment (FOMC文本情绪得分)
    触发: VIX 252日 Z-Score > 2.5 且 跨资产波动率双双回落(diff<0且下穿3日均值) 且 FOMC情绪5日边际变化 > 0.1
    输出: 脉冲信号 [-1.0, 1.0]。+1.0表示极度恐慌衰竭且鸽派确认看多，-1.0表示极度贪婪破灭且鹰派确认看空。
    """

    def __init__(self):
        self.name = 'unstructured_vol_crowding_reversal_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查所需数据字段是否存在
        required_cols = ['vixcls', 'gvzcls', 'fomc_sentiment']
        if not set(required_cols).issubset(data.columns):
            return pd.Series(0.0, index=data.index)

        # 前向填充处理非交易日或更新频率导致的缺失值
        df = data[required_cols].ffill()
        vix = df['vixcls']
        gvz = df['gvzcls']
        fomc = df['fomc_sentiment']

        # 初始化信号序列 (严格遵守零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)

        # --- 铁律2: 二阶导数确认 (Anti-Catch-Falling-Knife) ---
        # 1. 计算长周期波动率极值 (252日交易年)
        vix_mean = vix.rolling(window=252, min_periods=126).mean()
        vix_std = vix.rolling(window=252, min_periods=126).std()
        vix_zscore = (vix - vix_mean) / vix_std

        # 2. 衰竭确认: 跨资产波动率(美股+黄金)必须同步下降且低于3日平滑均线
        vix_falling = (vix.diff() < 0) & (vix < vix.rolling(window=3).mean())
        gvz_falling = (gvz.diff() < 0) & (gvz < gvz.rolling(window=3).mean())

        # 3. 破位确认 (用于做空场景): 波动率从低位开始同步抬头
        vix_rising = (vix.diff() > 0) & (vix > vix.rolling(window=3).mean())
        gvz_rising = (gvz.diff() > 0) & (gvz > gvz.rolling(window=3).mean())

        # --- 铁律3: 边际变化 (Marginal Change Only) ---
        # FOMC情绪得分呈低频阶梯状，使用5日滑动窗口(1个交易周)捕捉会议日后的情绪突变脉冲
        fomc_diff_5d = fomc.diff(periods=5)
        
        # 阈值0.1代表在[-1, 1]区间内至少发生5%的实质性NLP态度反转，过滤无意义噪音
        is_dovish_shock = fomc_diff_5d > 0.1
        is_hawkish_shock = fomc_diff_5d < -0.1

        # --- 触发核心逻辑 ---
        # 多头脉冲: 宏观极度恐慌(Z>2.5) + 全面恐慌退潮(波动率二阶导<0) + 美联储放鸽(文本情绪边际突变)
        long_trigger = (vix_zscore > 2.5) & vix_falling & gvz_falling & is_dovish_shock

        # 空头脉冲: 宏观极度贪婪(Z<-2.0) + 波动率低位反扑(波动率二阶导>0) + 美联储放鹰(文本情绪边际突变)
        short_trigger = (vix_zscore < -2.0) & vix_rising & gvz_rising & is_hawkish_shock

        # 赋值并只在触发日输出脉冲
        signal[long_trigger] = 1.0
        signal[short_trigger] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"