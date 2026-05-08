import numpy as np
import pandas as pd

class PanicExhaustionCrossFactor:
    """恐慌交叉与均值回归 (panic_mean_reversion/nonlinear)

    逻辑: 结合股市恐慌(VIX)与信用市场恐慌(高收益债利差)双重维度。当两者同时处于历史高位(Z-Score极值)且今日开始回落(转跌且低于3日均值)时, 确认恐慌衰竭, 输出强烈看多信号(+1.0); 当两者在非极值区突然5日内大幅飙升且当日还在上涨时, 确认恐慌情绪爆发, 输出看空信号(-1.0)。
    数据: [vixcls, bamlh0a0hym2]
    输出: [+1.0 恐慌极值衰竭抄底, -1.0 恐慌初期爆发破位]
    触发条件: [买入: 双Z-Score高位且均转跌破3日均线; 卖出: 5日大幅飙升且处于中低位且当日仍在上涨。预期Trigger Rate 5-15%]
    """

    def __init__(self, vix_z_threshold=1.5, hy_z_threshold=1.0, vix_surge=0.20, hy_surge=0.05):
        self.name = 'panic_exhaustion_cross'
        # 极值判断的Z-Score阈值
        self.vix_z_threshold = vix_z_threshold
        self.hy_z_threshold = hy_z_threshold
        # 起爆恶化判断的5日变化率阈值
        self.vix_surge = vix_surge
        self.hy_surge = hy_surge

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'bamlh0a0hym2']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 填充缺失值处理
        df = data[required_cols].ffill()
        vix = df['vixcls']
        hy = df['bamlh0a0hym2']
        
        # 计算 252日 Z-Score, 衡量长期极端偏离
        vix_mean_252 = vix.rolling(window=252, min_periods=60).mean()
        vix_std_252 = vix.rolling(window=252, min_periods=60).std()
        vix_z = (vix - vix_mean_252) / (vix_std_252 + 1e-8)
        
        hy_mean_252 = hy.rolling(window=252, min_periods=60).mean()
        hy_std_252 = hy.rolling(window=252, min_periods=60).std()
        hy_z = (hy - hy_mean_252) / (hy_std_252 + 1e-8)
        
        # 计算动量回落(二阶导数防接飞刀), 必须当日下跌且跌破3日均值
        vix_3ma = vix.rolling(window=3, min_periods=1).mean()
        hy_3ma = hy.rolling(window=3, min_periods=1).mean()
        
        vix_exhausted = (vix.diff(1) < 0) & (vix < vix_3ma)
        hy_exhausted = (hy.diff(1) < 0) & (hy < hy_3ma)
        
        # 计算短期恐慌爆发率 (过去5日的变化)
        vix_ret_5 = vix / (vix.shift(5) + 1e-8) - 1.0
        hy_ret_5 = hy / (hy.shift(5) + 1e-8) - 1.0
        
        # 恐慌起爆的当日延续性判断
        vix_rising = vix.diff(1) > 0
        hy_rising = hy.diff(1) > 0
        
        # --- 脉冲触发逻辑 ---
        
        # 买入条件(+1.0): 双重极端恐慌状态 + 动量衰竭(见顶回落)
        buy_cond = (
            (vix_z > self.vix_z_threshold) & 
            (hy_z > self.hy_z_threshold) & 
            vix_exhausted & 
            hy_exhausted
        )
        
        # 卖出条件(-1.0): 低位或常态区的恐慌突发飙升(防止极度恐慌期间连续发出卖点而死在黎明前)
        sell_cond = (
            (vix_ret_5 > self.vix_surge) & 
            (hy_ret_5 > self.hy_surge) & 
            (vix_z < 1.0) & 
            (hy_z < 1.0) & 
            vix_rising & 
            hy_rising
        )
        
        # 生成脉冲信号
        signal.loc[buy_cond] = 1.0
        signal.loc[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(vix_z_threshold={self.vix_z_threshold}, hy_z_threshold={self.hy_z_threshold})"