import numpy as np
import pandas as pd

class PolicyPivotRetracementFactor:
    """政策预期趋势回调衰竭因子 (Policy Pivot Retracement)

    逻辑: 顺应美联储的中期政策趋势(63日)，在短期逆势冲击(5日)衰竭时入场。即在降息周期(利率下行)中，买入短期鹰派恐慌造成的超跌；在加息周期(利率上行)中，做空短期鸽派乐观造成的超买。不接飞刀，跟随主升浪。
    数据: dgs2 (对政策最敏感的短端利率), t10y2y (收益率曲线形态)
    触发: dgs2中期趋势(63日)与短期脉冲(5日Z-Score极值)反向 + 脉冲二阶衰竭(价格回落且跌破3日均线) + 曲线动量重新顺应趋势
    输出: +1.0 看多美债(脉冲持仓3天), -1.0 看空美债(脉冲持仓3天)
    """

    def __init__(self):
        self.name = 'policy_pivot_retracement'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 1. 核心边际变化 (Marginal Changes)
        # 中期趋势 (63个交易日，约3个月)
        dgs2_trend = dgs2.diff(63)
        
        # 短期逆向脉冲 (5日变化量)
        dgs2_pulse = dgs2.diff(5)
        
        # 使用 63 日滚动窗口计算短期脉冲的 Z-Score，自适应近期波动率
        pulse_mean = dgs2_pulse.rolling(63).mean()
        pulse_std = dgs2_pulse.rolling(63).std().replace(0, np.nan)
        z_pulse = (dgs2_pulse - pulse_mean) / pulse_std
        
        # 2. 衰竭与二阶导数条件 (Anti-Catch-Falling-Knife)
        dgs2_ma3 = dgs2.rolling(3).mean()
        dgs2_diff1 = dgs2.diff(1)
        
        # 鹰派恐慌(利率冲高)衰竭: 当日利率下跌 且 低于3日均线
        spike_exhausted = (dgs2_diff1 < 0) & (dgs2 < dgs2_ma3)
        
        # 鸽派狂欢(利率下杀)衰竭: 当日利率反弹 且 高于3日均线
        plunge_exhausted = (dgs2_diff1 > 0) & (dgs2 > dgs2_ma3)
        
        # 3. 收益率曲线动量确认 (Curve Momentum)
        # 3日变化量，消除单日噪音
        t10y2y_mom = t10y2y.diff(3)
        
        # 4. 交叉触发逻辑组装
        # Long TLT: 中期降息趋势 + 短期鹰派脉冲极值 + 鹰派脉冲衰竭 + 曲线重新牛陡
        long_cond = (
            (dgs2_trend < -0.1) &       # 中期利率下行 > 10 bps
            (z_pulse > 1.2) &           # 短期利率暴涨 (Z-Score > 1.2)
            spike_exhausted &           # 冲高动量破位衰竭
            (t10y2y_mom > 0.0)          # 曲线重新变陡
        )
        
        # Short TLT: 中期加息趋势 + 短期鸽派脉冲极值 + 鸽派脉冲衰竭 + 曲线重新熊平
        short_cond = (
            (dgs2_trend > 0.1) &        # 中期利率上行 > 10 bps
            (z_pulse < -1.2) &          # 短期利率暴跌 (Z-Score < -1.2)
            plunge_exhausted &          # 下杀动量破位衰竭
            (t10y2y_mom < 0.0)          # 曲线重新变平
        )
        
        # 生成基础脉冲信号
        raw_signal = pd.Series(0.0, index=data.index)
        raw_signal[long_cond] = 1.0
        raw_signal[short_cond] = -1.0
        
        # 严格执行三大铁律中的 Zero Sleep 机制
        # 保持脉冲持续 3 天，以满足 5%-15% 的 Trigger Rate 目标
        signal_ffill = raw_signal.replace(0.0, np.nan).ffill(limit=2)
        signal = signal_ffill.fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"