import numpy as np
import pandas as pd

class CreditEquityPanicExhaustionFactor:
    """信用与权益恐慌极值见顶衰竭因子 (panic_mean_reversion/nonlinear)

    逻辑: 极度恐慌产生的是买点，但绝对不能接飞刀！当权益恐慌(VIX)或信用恐慌(HY OAS)达到年度高位(Z-Score > 1.5)，且今日两者双双录得下降(VIX.diff() < 0 且 利差收窄)时，确认恐慌二阶导转负(衰竭)，输出看多(+1.0)。若两者处于轻度恐慌(Z-Score在0.5至1.5间)且今日双双抬升，属于钝刀割肉的持续恶化趋势，输出看空(-1.0)。
    数据: [vixcls, bamlh0a0hym2]
    输出: +1.0(极度恐慌见顶衰竭，抄底买入), -1.0(轻度恐慌持续发酵，避险抛售), 0.0(常态休眠)
    触发条件: 1.5倍标准差极值交叉 + 双维度动量衰竭过滤，预期Trigger Rate在5%-15%区间
    """

    def __init__(self):
        self.name = 'credit_equity_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 所需字段：VIX波动率, 美银高收益债期权调整利差(衡量信用恐慌)
        required_cols = ['vixcls', 'bamlh0a0hym2']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        vix = data['vixcls'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()
        
        # 计算 252 个交易日（约1年）的滚动 Z-Score，反映相对所处历史位置
        lookback = 252
        vix_mean = vix.rolling(window=lookback, min_periods=60).mean()
        vix_std = vix.rolling(window=lookback, min_periods=60).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)
        
        hy_mean = hy_spread.rolling(window=lookback, min_periods=60).mean()
        hy_std = hy_spread.rolling(window=lookback, min_periods=60).std()
        hy_z = (hy_spread - hy_mean) / (hy_std + 1e-8)
        
        # 计算今日的边际变化 (防接飞刀的二阶导数特征)
        vix_diff = vix.diff()
        hy_diff = hy_spread.diff()
        
        # 1. 脉冲买入信号(+1.0)：极度恐慌 + 见顶衰竭
        # 经济学含义：权益或信用市场至少有一方出现极端流动性挤兑(Z>1.5)，且今日两者恐慌情绪同步回落
        extreme_panic = (vix_z > 1.5) | (hy_z > 1.5)
        panic_exhaustion = (vix_diff < 0) & (hy_diff < 0)
        buy_cond = extreme_panic & panic_exhaustion
        
        # 2. 脉冲卖出信号(-1.0)：轻微恐慌 + 持续发酵
        # 经济学含义：两个市场都跨过了安全线(Z>0.5)但未到极值，且今日仍在同步恶化(钝刀割肉)
        mild_panic = (vix_z > 0.5) & (vix_z <= 1.5) & (hy_z > 0.5) & (hy_z <= 1.5)
        panic_rising = (vix_diff > 0) & (hy_diff > 0)
        sell_cond = mild_panic & panic_rising
        
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"