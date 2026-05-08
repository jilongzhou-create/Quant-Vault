import numpy as np
import pandas as pd

class PolicyPivotPulseFactor:
    """Policy Pivot & Liquidity Impulse (nonlinear)

    逻辑: 捕捉美联储预期的突变(鸽派/鹰派转向)以及极端恐慌衰竭。通过2年期美债收益率与收益率曲线陡峭度的动量变化，精准定位美股的流动性拐点。
    数据: dgs2, t10y2y, vixcls, fomc_sentiment
    输出: [-1.0, 1.0]. +1.0 看多(鸽派突变/曲线牛陡/恐慌极值回落); -1.0 看空(鹰派突变/曲线熊平).
    触发条件: DGS2与T10Y2Y的5日动量Z-Score双重突破1.25标准差，或VIX极值(2.5标准差)后单日见顶回落，或FOMC文本情绪逆转。预期Trigger Rate控制在8%~12%。
    """

    def __init__(self):
        self.name = 'policy_pivot_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 必须的核心宏观底座数据
        req_cols = ['dgs2', 't10y2y']
        if not all(c in data.columns for c in req_cols):
            return signal
            
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 计算差分的5日动量，代表政策预期的短期剧变 (而非绝对水位)
        dgs2_5d = dgs2.diff(5)
        t10y2y_5d = t10y2y.diff(5)
        
        # 动态Z-Score适应不同波动率环境 (防魔法数字，避免在长期高息或低息时代信号失效)
        roll_mean_dgs2 = dgs2_5d.rolling(window=252, min_periods=63).mean()
        roll_std_dgs2 = dgs2_5d.rolling(window=252, min_periods=63).std()
        dgs2_z = (dgs2_5d - roll_mean_dgs2) / (roll_std_dgs2 + 1e-6)
        
        roll_mean_t10y = t10y2y_5d.rolling(window=252, min_periods=63).mean()
        roll_std_t10y = t10y2y_5d.rolling(window=252, min_periods=63).std()
        t10y2y_z = (t10y2y_5d - roll_mean_t10y) / (roll_std_t10y + 1e-6)
        
        # 基础条件：短端利率与收益率曲线的动量共振
        # 鸽派流动性释放 (Bull Steepener): 短端急降 + 曲线急陡
        bull_pivot = (dgs2_z < -1.25) & (t10y2y_z > 1.25)
        
        # 鹰派流动性收紧 (Hawkish Flattener): 短端急升 + 曲线变平/倒挂加深
        bear_pivot = (dgs2_z > 1.25) & (t10y2y_z < -1.25)
        
        # 引入 VIX 过滤防接飞刀与捕捉恐慌反转
        has_vix = 'vixcls' in data.columns
        if has_vix:
            vix = data['vixcls'].ffill()
            vix_5d = vix.diff(5)
            
            roll_mean_vix = vix.rolling(window=252, min_periods=63).mean()
            roll_std_vix = vix.rolling(window=252, min_periods=63).std()
            vix_level_z = (vix - roll_mean_vix) / (roll_std_vix + 1e-6)
            
            roll_mean_vix5d = vix_5d.rolling(window=252, min_periods=63).mean()
            roll_std_vix5d = vix_5d.rolling(window=252, min_periods=63).std()
            vix_5d_z = (vix_5d - roll_mean_vix5d) / (roll_std_vix5d + 1e-6)
            
            # 过滤1：真正的鸽派转向中，VIX不应处于暴力飙升状态 (避免买在股债双杀的主跌浪)
            bull_pivot = bull_pivot & (vix_5d_z < 1.0)
            
            # 过滤2：鹰派收水不能做空在恐慌极值点 (极度恐慌时常伴随强劲反抽)
            bear_pivot = bear_pivot & (vix_level_z < 1.5)
            
        # 赋值基础信号（由于使用 .diff(5)，这些状态自然会维持 2-4 天，形成完美的宽脉冲）
        signal[bear_pivot] = -1.0
        signal[bull_pivot] = 1.0
        
        # 附加模块：FOMC情绪边际跳跃 (阶梯低频数据的边际变化铁律)
        if 'fomc_sentiment' in data.columns:
            fomc = data['fomc_sentiment'].ffill()
            fomc_diff = fomc.diff(1)
            
            # 情绪从悲观突然跳升至乐观 (跳跃幅度>0.25)
            fomc_dovish_shock = (fomc.shift(1) < 0) & (fomc_diff > 0.25)
            # 情绪从乐观突然跳降至悲观
            fomc_hawkish_shock = (fomc.shift(1) > 0) & (fomc_diff < -0.25)
            
            # 对单日事件展期2天，确保捕获脉冲内的交易窗口
            signal[fomc_hawkish_shock.rolling(2).max() == 1] = -1.0
            signal[fomc_dovish_shock.rolling(2).max() == 1] = 1.0
            
        # 附加模块：二阶导数铁律 - 极度恐慌衰竭 (无视其他条件，优先级最高)
        if has_vix:
            # 逻辑：VIX处于历史极端高位(>2.5 Z-Score) 且 当天VIX开始回落(.diff(1)<0) -> 此时才是安全抄底点
            panic_exhaustion = (vix_level_z > 2.5) & (vix.diff(1) < 0)
            signal[panic_exhaustion.rolling(2).max() == 1] = 1.0
            
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"