import numpy as np
import pandas as pd

class FinancialStressVixExhaustionFactor:
    """金融压力与VIX恐慌衰竭交叉因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合低频的圣路易斯金融压力指数(STLFSI)判定系统性压力环境，与高频的VIX波动率捕捉情绪极值。在宏观金融高压区，当VIX创极值后跌破3日均线，产生强烈的恐慌见顶衰竭抄底脉冲；在平时环境中，如果VIX突发暴涨破坏平静形态，则产生初期恶化的看空脉冲。
    数据: stlfsi4 (金融压力指数), vixcls (VIX隐含波动率)
    输出: +1.0 表示系统恐慌见顶衰竭(强烈看多脉冲), -1.0 表示平静期突发流动性恶化/恐慌飙升(看空脉冲)
    触发条件: 抄底要求STLFSI Z-Score>0.5 且 VIX Z-Score>1.2 且 VIX回落；做空要求VIX单日突增>3.0且尚未进入极度恐慌。预期Trigger Rate在 5% 到 15% 之间。
    """

    def __init__(self):
        self.name = 'financial_stress_vix_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 默认信号输出全 0.0 的 Series
        signal = pd.Series(0.0, index=data.index)
        
        # 检查是否包含所需数据列
        if 'stlfsi4' not in data.columns or 'vixcls' not in data.columns:
            return signal
            
        # 前向填充处理低频和缺失数据 (如STLFSI为周频)
        stlfsi = data['stlfsi4'].ffill()
        vix = data['vixcls'].ffill()
        
        # 避免全部为空值的情况
        if stlfsi.isna().all() or vix.isna().all():
            return signal
            
        # 1. 计算宏观金融压力状态 (252个交易日约等于一年窗口)
        # stlfsi4 虽然是低频，ffill 展开成日频后进行滚动统计可以平稳映射压力周期
        stlfsi_mean = stlfsi.rolling(window=252, min_periods=60).mean()
        stlfsi_std = stlfsi.rolling(window=252, min_periods=60).std()
        # 计算 Z-Score，防除零异常
        stlfsi_z = (stlfsi - stlfsi_mean) / stlfsi_std.replace(0, np.nan)
        
        # 2. 计算VIX的局部恐慌极值状态 (126个交易日约等于半年窗口)
        vix_mean = vix.rolling(window=126, min_periods=30).mean()
        vix_std = vix.rolling(window=126, min_periods=30).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        # 3. 计算边际变化与动量 (核心二阶导数约束，防接飞刀)
        vix_diff_1 = vix.diff(1)
        vix_ma_3 = vix.rolling(window=3).mean()
        
        # 4. 买入脉冲条件 (极度恐慌状态 + 明确见顶回落衰竭)
        # 必须满足: 宏观偏紧(Z>0.5) 且 情绪极度恐慌(Z>1.2) 且 VIX当日收阴且跌破3日均线
        is_high_stress = stlfsi_z > 0.5
        is_vix_panic = vix_z > 1.2
        is_vix_exhausted = (vix_diff_1 < 0) & (vix < vix_ma_3)
        
        buy_condition = is_high_stress & is_vix_panic & is_vix_exhausted
        
        # 5. 卖出脉冲条件 (恐慌恶化初期的流动性突变)
        # 必须满足: 宏观压力尚未见顶(Z<0.8) 且 尚未处于极端高位(Z<1.0) 且 VIX单日异常暴涨超3点
        # 这一条件确保只在下杀主跌浪的第一时间触发看空，禁止高VIX时接飞刀
        is_normal_stress = stlfsi_z < 0.8
        not_already_panic = vix_z < 1.0
        vix_spike = vix_diff_1 > 3.0
        
        sell_condition = is_normal_stress & not_already_panic & vix_spike
        
        # 输出脉冲信号
        signal[buy_condition] = 1.0
        signal[sell_condition] = -1.0
        
        # 确保数据起步时的 NaN 被处理为 0.0
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"