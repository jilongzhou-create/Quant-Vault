import numpy as np
import pandas as pd

class PanicMeanReversionYieldCrossFactor:
    """panic_mean_reversion/nonlinear

    逻辑: 恐慌极值叠加衰竭与收益率曲线陡峭化产生强烈买点；轻度恐慌叠加曲线平坦化产生卖点。
    数据: vixcls, bamlh0a0hym2, t10y2y
    输出: 1.0 (抄底买入), -1.0 (趋势恶化看空), 0.0 (常态)
    触发条件: 恐慌高位且VIX与信用利差实质性回落且曲线变陡时看多；低位缓慢上升且曲线变平时看空，预期 Trigger Rate 5% 到 15%。
    """

    def __init__(self):
        self.name = 'panic_mean_reversion_yield_cross'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        required_cols = ['vixcls', 'bamlh0a0hym2', 't10y2y']
        
        # 检查所需数据字段是否齐全
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        vix = data['vixcls'].ffill()
        hy = data['bamlh0a0hym2'].ffill()
        curve = data['t10y2y'].ffill()

        # 计算126日滚动Z-Score，用于识别动态恐慌环境
        # 限制 std 下限防止除以 0
        vix_mean = vix.rolling(126).mean()
        vix_std = vix.rolling(126).std().clip(lower=0.001)
        hy_mean = hy.rolling(126).mean()
        hy_std = hy.rolling(126).std().clip(lower=0.001)

        vix_z = (vix - vix_mean) / vix_std
        hy_z = (hy - hy_mean) / hy_std

        # 计算边际动量变化 (一阶/多阶导数)
        vix_diff3 = vix.diff(3)
        vix_diff1 = vix.diff(1)
        hy_diff3 = hy.diff(3)
        curve_diff10 = curve.diff(10)

        # ---------------------------------------------------------------------
        # 多头信号: 极度恐慌 + 动量衰竭 + 宽松预期 (严格防接飞刀)
        # ---------------------------------------------------------------------
        # 1. 过去10天内，VIX或信用利差曾达到极端高位 (Z-Score > 1.5，统计学上的尾部事件)
        panic_regime = (vix_z.rolling(10).max() > 1.5) | (hy_z.rolling(10).max() > 1.5)
        
        # 2. 恐慌实质性衰竭: VIX 3日和1日均下滑, 信用利差收窄，且 VIX 跌破10日均线 (确认见顶并向下发散)
        exhaustion = (vix_diff3 < -0.5) & (vix_diff1 < 0.0) & (hy_diff3 < 0.0) & (vix < vix.rolling(10).mean())
        
        # 3. 流动性预期改善: 收益率曲线未进一步平坦化 (曲线变陡意味着短端利率预期下行，提供抄底流动性)
        policy_easing = curve_diff10 > 0.0
        
        buy_cond = panic_regime & exhaustion & policy_easing

        # ---------------------------------------------------------------------
        # 空头信号: 自满情绪 + 恐慌悄然蔓延 + 紧缩压力 (捕捉钝刀割肉的下跌波段)
        # ---------------------------------------------------------------------
        # 1. 常态/自满环境 (未处于极端恐慌中，防止被深V暴打)
        complacency = (vix_z < 1.0) & (hy_z < 1.0)
        
        # 2. 恐慌悄然蔓延: VIX 与 信用利差 均温和且持续地上升，并突破近期均线
        creeping_fear = (vix_diff3 > 0.5) & (hy_diff3 > 0.02) & (vix > vix.rolling(5).mean()) & (hy > hy.rolling(10).mean())
        
        # 3. 流动性紧缩: 收益率曲线实质性平坦化 (宏观资金面在抽水)
        hawkish = curve_diff10 < -0.01

        sell_cond = complacency & creeping_fear & hawkish
        
        # 剔除可能存在的重叠冲突
        buy_cond = buy_cond & ~sell_cond

        # 构建脉冲信号
        signal = pd.Series(0.0, index=data.index)
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0

        return signal.fillna(0.0).rename(self.name)

    def __repr__(self):
        return f"{self.__class__.__name__}()"