import numpy as np
import pandas as pd

class PanicMeanReversionNonlinearFactor:
    """恐慌均值回归与非线性特征交叉因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合VIX与高收益债信用利差(HY Spread)衡量系统性流动性恐慌。当股市波动与信用利差均达到极值且VIX二阶导由正转负时(恐慌见顶衰竭)，产生强抄底脉冲；当波动和利差小幅上穿警戒线时，视为钝刀割肉初期，产生看空脉冲。
    数据: vixcls, bamlh0a0hym2
    输出: 强看多(+1.0)表示恐慌衰竭的极佳买点，轻微恐慌恶化输出看空(-1.0)
    触发条件: 股市恐慌与信用利差双重高企且VIX跌破3日均值时触发买入，VIX短线Z-Score突破0.5且利差走阔时触发卖出。预期Trigger Rate: 8%-12%
    """

    def __init__(self):
        self.name = 'panic_mean_reversion_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 确保所需数据列存在
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()
        
        # ==========================================
        # 1. 计算系统性压力的长期与短期 Z-Score
        # ==========================================
        # 长期基准 (252日 = 1个交易年)
        vix_ma252 = vix.rolling(252).mean()
        vix_std252 = vix.rolling(252).std()
        vix_z_252 = (vix - vix_ma252) / vix_std252
        
        hy_ma252 = hy_spread.rolling(252).mean()
        hy_std252 = hy_spread.rolling(252).std()
        hy_z_252 = (hy_spread - hy_ma252) / hy_std252
        
        # 短期基准 (21日 = 1个交易月)
        vix_ma21 = vix.rolling(21).mean()
        vix_std21 = vix.rolling(21).std()
        vix_z_21 = (vix - vix_ma21) / vix_std21
        
        hy_ma21 = hy_spread.rolling(21).mean()
        hy_std21 = hy_spread.rolling(21).std()
        hy_z_21 = (hy_spread - hy_ma21) / hy_std21
        
        # ==========================================
        # 2. 买入逻辑 (防接飞刀: 极值 + 衰竭转折)
        # ==========================================
        # 极度恐慌条件: 波动率与信用利差至少在短期或长期达到统计学异常高位
        is_panic = (vix_z_252 > 1.5) | (vix_z_21 > 1.5)
        is_stress = (hy_z_252 > 1.0) | (hy_z_21 > 1.0)
        
        # 衰竭二阶导条件: 绝不能在VIX主升浪中买入! 必须等待恐慌边际回落
        # 当日下跌, 且价格低于过去3日均值, 代表动能完全扭转
        vix_reverting = (vix.diff(1) < 0) & (vix < vix.rolling(3).mean())
        
        buy_pulse = is_panic & is_stress & vix_reverting
        
        # ==========================================
        # 3. 卖出逻辑 (脉冲式看空: 钝刀割肉恶化初期)
        # ==========================================
        # 波动率刚刚突破短期危险阈值 (Z-Score 上穿 0.5)
        vix_worsening = (vix_z_21 > 0.5) & (vix_z_21.shift(1) <= 0.5)
        
        # 信用市场同步恶化, 说明非随机扰动 (利差在平均线以上且近期在上升)
        credit_worsening = (hy_z_21 > 0.0) & (hy_spread.diff(3) > 0)
        
        # 排除已经极度恐慌的接飞刀末期阶段
        not_extreme = (vix_z_21 < 1.5) & (vix_z_252 < 1.5)
        
        sell_pulse = vix_worsening & credit_worsening & not_extreme
        
        # ==========================================
        # 4. 生成脉冲信号
        # ==========================================
        signal[buy_pulse] = 1.0
        signal[sell_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"