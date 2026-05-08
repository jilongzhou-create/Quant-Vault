import numpy as np
import pandas as pd

class UnstructuredPolicyPivotPulseFactor:
    """非结构化政策转向脉冲因子 (policy_pivot/unstructured)

    逻辑: 捕捉由于央行声明文本(FOMC NLP)和新闻政策不确定性(EPU)突变所带来的宏观极值反转。当美联储声明情绪发生边际大幅转变(变鸽看多，变鹰看空)时，由于机构调仓需要时间，顺势释放极短期的脉冲信号。当极端的宏观政策不确定性(Z>2.5)终于见顶回落(衰竭)时，市场由于"利空出尽"出现均值回归的抄底反弹。
    数据: fomc_sentiment(FOMC声明NLP情绪得分), usepuindxd(美国经济政策不确定性文本指数)
    输出: +1.0 表示强烈的政策鸽派突变或极端政策恐慌衰竭(看多)；-1.0 表示政策鹰派突变(看空)。常态下为 0.0。
    触发条件: FOMC会议情绪边际跳升>0.25或跨0轴反转，或EPU恐慌指数Z-Score>2.5且开始回落。脉冲预期 Trigger Rate 约 8%-12%。
    """

    def __init__(self, pulse_window: int = 3, fomc_delta_threshold: float = 0.25, epu_zscore_threshold: float = 2.5):
        self.name = 'unstructured_policy_pivot_pulse'
        self.pulse_window = pulse_window
        self.fomc_delta_threshold = fomc_delta_threshold
        self.epu_zscore_threshold = epu_zscore_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 如果必要数据缺失，直接休眠返回0.0
        if 'fomc_sentiment' not in data.columns or 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        fomc = data['fomc_sentiment'].ffill()
        epu = data['usepuindxd'].ffill()
        
        if fomc.isna().all() or epu.isna().all():
            return pd.Series(0.0, index=data.index)

        # --------------------------------------------------------------------------------
        # 1. 货币政策 NLP 情绪突变脉冲 (边际变化铁律)
        # FOMC 阶梯数据，只在预期改变瞬间(T+1日)获取差分跳跃，绝不输出绝对值
        # --------------------------------------------------------------------------------
        fomc_delta = fomc.diff()
        
        # 鸽派突变：绝对情绪得分大幅边际改善(>=0.25)，或者跨越0轴(由鹰转鸽)且有明显的积极确认(>=0.15)
        dovish_cond = (fomc_delta >= self.fomc_delta_threshold) | \
                      ((fomc > 0.0) & (fomc.shift(1) <= 0.0) & (fomc_delta >= 0.15))
                      
        # 鹰派突变：绝对情绪得分大幅边际恶化(<=-0.25)，或者跨越0轴(由鸽转鹰)且有明显的负面确认(<=-0.15)
        hawkish_cond = (fomc_delta <= -self.fomc_delta_threshold) | \
                       ((fomc < 0.0) & (fomc.shift(1) >= 0.0) & (fomc_delta <= -0.15))
        
        fomc_signal = pd.Series(0.0, index=data.index)
        fomc_signal[dovish_cond] = 1.0
        fomc_signal[hawkish_cond] = -1.0
        
        # 将会议日的瞬时巨变脉冲，顺延 pulse_window 天，模拟短期机构情绪消化的买/卖动量窗口
        # 使用 replace 使得我们能平滑传递脉冲，limit = 窗口长度 - 1
        fomc_pulse = fomc_signal.replace(0.0, np.nan).ffill(limit=self.pulse_window - 1).fillna(0.0)

        # --------------------------------------------------------------------------------
        # 2. 宏观政策不确定性 NLP 情绪极值衰竭 (防飞刀二阶导铁律)
        # 基于新闻文本的不确定性指数。平时噪声多须忽略，只在极度恐慌期间出利空落地时抓反弹
        # --------------------------------------------------------------------------------
        # 计算 252 交易日滚动均值和标准差，获取局部 Z-Score
        epu_mean = epu.rolling(window=252, min_periods=126).mean()
        epu_std = epu.rolling(window=252, min_periods=126).std()
        epu_z = (epu - epu_mean) / (epu_std + 1e-6)
        
        # 抄底条件: 政策恐慌处于历史高位(Z > 2.5)，且动量衰竭(近两日不确定性连续回落, 二阶导数<0)
        # 这意味着靴子落地或政策安抚生效，标普500往往迎来长牛均值回归的抄底反弹
        epu_exhaustion_cond = (epu_z.shift(1) > self.epu_zscore_threshold) & (epu.diff(2) < 0.0)
        
        epu_pulse = pd.Series(0.0, index=data.index)
        epu_pulse[epu_exhaustion_cond] = 1.0
        
        # 为了脉冲不至于只有一天，给予极短的顺延 (总共持续2天)
        epu_pulse = epu_pulse.replace(0.0, np.nan).ffill(limit=1).fillna(0.0)

        # --------------------------------------------------------------------------------
        # 3. 信号合并与截断
        # --------------------------------------------------------------------------------
        final_signal = fomc_pulse + epu_pulse
        
        # 严格截断在 [-1.0, 1.0] 范围内，并且填补空值防止错误
        final_signal = final_signal.clip(-1.0, 1.0).fillna(0.0)
        final_signal.name = self.name
        
        return final_signal

    def __repr__(self):
        return f"{self.__class__.__name__}(pulse_window={self.pulse_window}, fomc_delta_thr={self.fomc_delta_threshold}, epu_z_thr={self.epu_zscore_threshold})"