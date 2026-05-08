import numpy as np
import pandas as pd

class VolatilityCreditCrossReversionFactor:
    """波动率与信用利差交叉反转因子 (panic_mean_reversion/nonlinear)

    逻辑: 恐慌与贪婪均具有极值回归属性。当中期违约风险(信用利差)与短期恐慌(VIX)同时达到统计学极端高位时，一旦VIX短期动量发生向下反转(死叉衰竭)，预示着市场恐慌已过极点，将迎来估值修复(强烈看多)；相反，当市场长时间极度平静被打破(金叉抬头)时看空美股恶化。
    数据: vixcls, bamlh0a0hym2
    输出: +1.0 看多(恐慌衰竭抄底), -1.0 看空(过度平静被打破), 常态 0.0
    触发条件: 满足极端高压或低压Z-Score条件，并且当日VIX Z-Score恰好穿越其5日均线产生单日脉冲。预期Trigger Rate: 5%~15%
    """

    def __init__(self):
        self.name = 'volatility_credit_cross_reversion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1&数据预检: 如果缺少核心字段，直接返回全0序列
        required_cols = ['vixcls', 'bamlh0a0hym2']
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)
                
        # 避免 nan 值影响计算，取宏观数据常用前向填充
        vix = data['vixcls'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()
        
        # 采用126日(半年)滚动窗口计算中期分布极值 (Z-Score)
        rolling_window = 126
        
        vix_mean = vix.rolling(window=rolling_window).mean()
        vix_std = vix.rolling(window=rolling_window).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std
        
        hy_mean = hy_spread.rolling(window=rolling_window).mean()
        hy_std = hy_spread.rolling(window=rolling_window).std().replace(0, np.nan)
        hy_z = (hy_spread - hy_mean) / hy_std
        
        # 获取极短期的动量基准 (5日即一周)
        short_window = 5
        vix_z_short_mean = vix_z.rolling(window=short_window).mean()
        
        # === 铁律7 (二阶导数铁律): 防接飞刀，捕捉抄底买点 (+1.0) ===
        # 条件A：极度恐慌确认 - 过去一周内VIX曾达到极端水平 (Z > 1.5)
        high_panic_vix = vix_z.rolling(window=5).max() > 1.5
        # 条件B：信用紧缩确认 - 高收益债利差(违约预期)仍处于高压 (Z > 1.0)
        high_panic_hy = hy_z > 1.0
        # 条件C：恐慌衰竭(转折) - VIX在今日向下穿越5日短期均线 (恐慌明确退潮的瞬间)
        vix_exhaustion = (vix_z < vix_z_short_mean) & (vix_z.shift(1) >= vix_z_short_mean.shift(1))
        
        buy_pulse = high_panic_vix & high_panic_hy & vix_exhaustion
        
        # === 铁律9 (SPY长牛均值回归): 捕捉过度平静被打破的看空卖点 (-1.0) ===
        # 条件A：极度贪婪/平静确认 - 过去一周内VIX处于极低位 (Z < -1.0)
        extreme_calm_vix = vix_z.rolling(window=5).min() < -1.0
        # 条件B：信用极度宽松确认 (Z < -0.5)
        extreme_calm_hy = hy_z < -0.5
        # 条件C：趋势恶化(转折) - VIX在今日向上穿越5日短期均线 (平静被打破的瞬间)
        vix_surge = (vix_z > vix_z_short_mean) & (vix_z.shift(1) <= vix_z_short_mean.shift(1))
        
        sell_pulse = extreme_calm_vix & extreme_calm_hy & vix_surge
        
        # === 铁律6 (零值休眠铁律): 输出纯粹脉冲 ===
        signal = pd.Series(0.0, index=data.index)
        signal[buy_pulse] = 1.0
        signal[sell_pulse] = -1.0
        
        # 清理由于均线尚未初始化(前期)带来的 NaN，保证输出干净
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"