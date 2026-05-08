import numpy as np
import pandas as pd

class UnstructuredPolicyUncertaintyOptionsFactor:
    """非结构化政策不确定性与期权波动率共振因子 (unstructured/options)

    逻辑: 结合经济政策不确定性指数(EPU)与期权隐含波动率(VIX)及收益率曲线形态(T10Y2Y)。
         为了避免与其他波动率因子重合，完全抛弃常规的多资产波动率(如VIX+黄金)逻辑。
         当宏观政策不确定性(EPU)出现极端跳跃(Z-Score > 2.5)，说明新闻文本端的恐慌指标爆表。
         此时要求期权市场(VIX)拒绝创新高并开始回落衰竭(VIX < 3日均线)，防范主跌浪接飞刀。
         同时收益率曲线剧烈变陡(T10Y2Y 动量 > 0，Bull Steepening)，确认债券市场Price-in美联储将立即降息救市。
         此时触发多头狙击脉冲。当政策不确定性断崖解除且曲线走平时触发空头。
    数据: usepuindxd (文本不确定性), vixcls (期权隐含波动率), t10y2y (收益率曲线利差)
    触发: 过去3日内 EPU diff Z-Score > 2.5 + 当日 VIX < 3MA + 当日 T10Y2Y diff > 0 -> 脉冲 +1.0
    输出: [-1.0, 1.0] 的多空狙击级脉冲信号。
    """

    def __init__(self):
        self.name = 'unstructured_policy_uncertainty_options'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 返回pd.Series(0.0) 如果所需列缺失，防止程序崩溃
        required_cols = ['usepuindxd', 'vixcls', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        # 向前填充处理缺失值
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # 铁律3: 绝对禁止使用绝对值，必须使用边际变化(动量计算)
        epu_diff = epu.diff(3)
        
        # 滚动126天(约半年)计算局部Z-Score，识别新闻端的极端脉冲事件
        epu_mean = epu_diff.rolling(126, min_periods=20).mean()
        epu_std = epu_diff.rolling(126, min_periods=20).std().replace(0, np.nan)
        epu_z = (epu_diff - epu_mean) / epu_std

        # 铁律1 & 2条件1: 识别极值突变 (给极端事件3天的观察窗口期)
        recent_epu_spike = (epu_z > 2.5).rolling(3, min_periods=1).max() > 0
        recent_epu_plunge = (epu_z < -2.5).rolling(3, min_periods=1).max() > 0

        # 铁律2条件2: 衰竭信号防接飞刀 (要求当天必须发生衰竭回落)
        vix_exhaustion_long = vix < vix.rolling(3).mean()
        vix_exhaustion_short = vix > vix.rolling(3).mean()

        # 债市确认: 收益率曲线动量变化 (只看变化，不看绝对是否倒挂)
        # Bull Steepening 确认短端剧烈下行（降息预期升温）
        curve_steepening = t10y2y.diff(3) > 0
        # Bear Flattening 确认短端坚挺（加息/紧缩预期升温）
        curve_flattening = t10y2y.diff(3) < 0

        # 脉冲触发条件 (极端新闻冲击 + 期权市场恐慌衰竭 + 利率市场定价确认)
        raw_long = recent_epu_spike & vix_exhaustion_long & curve_steepening
        raw_short = recent_epu_plunge & vix_exhaustion_short & curve_flattening

        # 铁律1: 延长有效脉冲生命周期5天，确保触发频率稳定在 5%-15% 的目标区间
        long_pulse = raw_long.astype(float).rolling(5, min_periods=1).max()
        short_pulse = raw_short.astype(float).rolling(5, min_periods=1).max()

        # 零值休眠初始化
        signal = pd.Series(0.0, index=data.index)
        
        # 信号赋值，确保极端信号出现后为+1/-1，其余常态均为0
        signal[short_pulse == 1.0] = -1.0
        signal[long_pulse == 1.0] = 1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"