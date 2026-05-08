import numpy as np
import pandas as pd

class EpuVixPolicyPivotFactor:
    """经济政策不确定性与期权恐慌衰竭因子 (unstructured/options)

    逻辑: 结合基于新闻文本的经济政策不确定性(EPU)与期权隐含波动率(VIX)，捕捉宏观极度恐慌时刻。
          利用二阶导数等待恐慌衰竭，并结合2年期美债(DGS2)的边际变化判断恐慌性质：
          若恐慌伴随短端利率飙升(加息恐慌)，衰竭时预示收益率见顶，看多美债(脉冲+1.0)；
          若恐慌伴随短端利率暴跌(衰退恐慌)，衰竭时预示避险情绪消退，看空美债(脉冲-1.0)。
    数据: usepuindxd (EPU), vixcls (VIX), dgs2 (2Y国债收益率)
    触发: EPU与VIX的5日动量Z-Score同时 > 1.5，且在随后3天内开始双双回落(<3日均值)时触发。
    输出: 狙击手级脉冲信号，非触发日严格为 0.0。
    """

    def __init__(self):
        self.name = 'epu_vix_policy_pivot_factor'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0.0 Series (铁律1: 零值休眠)
        signal = pd.Series(0.0, index=data.index)

        # 数据完整性检查
        req_cols = ['usepuindxd', 'vixcls', 'dgs2']
        for col in req_cols:
            if col not in data.columns:
                return signal

        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        dgs2 = data['dgs2'].ffill()

        # 铁律3: 边际变化 (Marginal Change Only)
        # 5日动量代表单周的剧烈情绪边际变化
        epu_diff = epu.diff(5).fillna(0)
        vix_diff = vix.diff(5).fillna(0)
        
        # 锚定宏观政策预期的方向：使用10天(双周)的短端利率累积变化过滤噪音
        dgs2_diff = dgs2.diff(10).fillna(0)

        # 计算 252 日(1年)滚动 Z-Score，设定最小观测期 21 天(1个月)
        epu_mean = epu_diff.rolling(window=252, min_periods=21).mean()
        epu_std = epu_diff.rolling(window=252, min_periods=21).std().replace(0, np.nan)
        epu_z = (epu_diff - epu_mean) / epu_std

        vix_mean = vix_diff.rolling(window=252, min_periods=21).mean()
        vix_std = vix_diff.rolling(window=252, min_periods=21).std().replace(0, np.nan)
        vix_z = (vix_diff - vix_mean) / vix_std

        # 铁律1: 极端脉冲 (Sniper Pulse)
        # EPU与VIX的单周动量同时异动 (Z-Score > 1.5 约代表单一指标的 top 7% 尾部事件，叠加后属于极低概率的共振恐慌)
        is_panic = (epu_z > 1.5) & (vix_z > 1.5)
        
        # 允许该共振恐慌在过去3天内发生 (为衰竭留出极短的时间窗口)
        recent_panic = is_panic.rolling(window=3, min_periods=1).max() > 0

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 必须等待指标明确衰竭：双双跌破3日均线，且日内边际(diff)仍在回落
        epu_exhaustion = (epu < epu.rolling(window=3).mean()) & (epu.diff(1) < 0)
        vix_exhaustion = (vix < vix.rolling(window=3).mean()) & (vix.diff(1) < 0)
        is_exhausted = epu_exhaustion & vix_exhaustion

        # 依据前瞻政策指标(dgs2)判断恐慌性质及衰竭后的资产方向
        # > 0.05 (5 bps): 短端利率呈上行趋势，属于加息恐慌/通胀恐慌
        is_hawkish = dgs2_diff > 0.05
        # < -0.05 (-5 bps): 短端利率呈下行趋势，属于衰退恐慌/降息预期
        is_dovish = dgs2_diff < -0.05

        # 组合脉冲信号
        cond_buy = recent_panic & is_exhausted & is_hawkish
        cond_sell = recent_panic & is_exhausted & is_dovish

        signal[cond_buy] = 1.0
        signal[cond_sell] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"