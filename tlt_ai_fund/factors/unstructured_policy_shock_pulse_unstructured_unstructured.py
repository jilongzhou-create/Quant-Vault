import numpy as np
import pandas as pd

class UnstructuredPolicyShockPulseFactor:
    """非结构化政策冲击脉冲因子 (unstructured/unstructured)

    逻辑: 捕捉NLP非结构化数据衍生的政策不确定性(EPU)与联储情绪(FOMC Sentiment)的预期极值与衰竭。
         当经济政策不确定性在短期内极速飙升后见顶回落，或联储文本释放超预期的边际鸽派信号并企稳时，
         市场避险需求与降息预期共振，催生美债多头脉冲。反之做空。必须等待极端动能开始衰竭才触发信号，防接飞刀。
    数据: usepuindxd (经济政策不确定性), fomc_sentiment (FOMC文本情绪得分)
    触发: 边际动量 Z-Score > 1.5 (极值) 且 短期绝对水平相对均值/前值发生逆转 (二阶导衰竭)
    输出: +1.0 看多美债 (避险买盘/鸽派), -1.0 看空美债 (风险偏好回归/鹰派)，常态严格为 0.0
    """

    def __init__(self):
        self.name = 'unstructured_policy_shock_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        has_epu = 'usepuindxd' in data.columns
        has_fomc = 'fomc_sentiment' in data.columns
        
        if not has_epu and not has_fomc:
            return signal

        long_cond = pd.Series(False, index=data.index)
        short_cond = pd.Series(False, index=data.index)

        # 模块1: 经济政策不确定性 (EPU) 恐慌极值反转脉冲
        if has_epu:
            epu = data['usepuindxd'].ffill()
            
            # 边际变化铁律: 捕捉短期动能突变 (10日变化)
            epu_mom = epu.diff(10)
            epu_roll_mean = epu_mom.rolling(window=252, min_periods=60).mean()
            epu_roll_std = epu_mom.rolling(window=252, min_periods=60).std() + 1e-6
            epu_zscore = (epu_mom - epu_roll_mean) / epu_roll_std
            
            # 二阶导数铁律: 必须等恐慌动能衰竭、趋势明确拐头才抄底
            epu_exhaustion_up = epu < epu.rolling(5).mean()    # 暴涨后均线压制回落
            epu_exhaustion_down = epu > epu.rolling(5).mean()  # 暴跌后均线支撑反弹
            
            # 触发: 极高不确定性开始衰退 -> 资金避险买入长债
            epu_long = (epu_zscore > 1.5) & epu_exhaustion_up
            # 触发: 极度平静期被打破 -> 风险溢价重估抛售长债
            epu_short = (epu_zscore < -1.5) & epu_exhaustion_down
            
            long_cond = long_cond | epu_long
            short_cond = short_cond | epu_short

        # 模块2: 央行情绪 (FOMC Sentiment) 预期跳跃企稳脉冲
        if has_fomc:
            fomc = data['fomc_sentiment'].ffill()
            
            # 边际变化铁律: 阶梯数据必须使用 diff 捕捉会议落地的跳跃
            fomc_mom = fomc.diff(5)
            fomc_roll_mean = fomc_mom.rolling(window=252, min_periods=60).mean()
            fomc_roll_std = fomc_mom.rolling(window=252, min_periods=60).std() + 1e-6
            fomc_zscore = (fomc_mom - fomc_roll_mean) / fomc_roll_std
            
            # 二阶导数铁律: 跳跃发生的动量差值(加速度)不再创新高，代表信息被市场充分吸收
            mom_accel = fomc_mom.diff(1)
            fomc_exhaustion_dove = mom_accel <= 0  # 鸽派脉冲停止加速
            fomc_exhaustion_hawk = mom_accel >= 0  # 鹰派脉冲停止下探
            
            # 触发: 极度边际转鸽 + 冲击企稳 -> 顺势做多
            fomc_long = (fomc_zscore > 1.5) & fomc_exhaustion_dove & (fomc_mom > 0)
            # 触发: 极度边际转鹰 + 冲击企稳 -> 顺势做空
            fomc_short = (fomc_zscore < -1.5) & fomc_exhaustion_hawk & (fomc_mom < 0)
            
            long_cond = long_cond | fomc_long
            short_cond = short_cond | fomc_short
            
        # 零值休眠铁律: 脉冲输出，其余常态严格保持 0.0
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        # 处理逻辑冲突的极端重合日，重置为 0
        conflict = long_cond & short_cond
        signal[conflict] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"