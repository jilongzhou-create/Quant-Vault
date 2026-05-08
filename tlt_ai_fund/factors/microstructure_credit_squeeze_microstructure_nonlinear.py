import numpy as np
import pandas as pd

class RateExpectationShockFactor:
    """短期利率预期冲击与衰竭因子 (Microstructure/Nonlinear)

    逻辑: FICC短端利率(dtb3)的剧烈变化代表了市场对美联储货币政策(加息/降息)的极端预期冲击。当短端利率的边际变化(3日动量)达到统计学极值，且开始向均值衰竭时，意味着单边的政策恐慌/狂热已经过度定价(Priced In)。此时收益率曲线(t10y2y)的同步变形可作为交叉验证。反向做多/做空美债(TLT)捕捉预期的均值回归。脉冲信号能精准狙击预期扭转的瞬间。
    数据: dtb3 (3个月美债收益率, 代表政策预期), t10y2y (10年-2年期限利差, 代表曲线结构)
    触发: dtb3的3日动量 Z-Score的绝对值 > 1.2，且动量开始衰竭(突破3日均值)，且得到期限利差对应方向的交叉验证。
    输出: +1.0 (加息恐慌极致且衰竭，做多TLT) / -1.0 (降息狂热极致且衰竭，做空TLT)
    """

    def __init__(self):
        self.name = 'rate_expectation_shock_micro_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号为全 0.0 (遵守铁律1: 零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        # 基础列检查，缺失则返回全0
        if 'dtb3' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        dtb3 = data['dtb3'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change)
        # 使用3日差分计算短端政策预期的脉冲动量，过滤单日标售噪音
        velocity = dtb3.diff(3)
        
        # 计算252日(1年)滚动Z-Score，衡量预期冲击的极端程度
        mean_252 = velocity.rolling(window=252, min_periods=60).mean()
        std_252 = velocity.rolling(window=252, min_periods=60).std()
        std_252 = std_252.replace(0, np.nan)
        z_score = (velocity - mean_252) / std_252
        
        # 收益率曲线的3日边际变形，用于交叉验证
        curve_change = t10y2y.diff(3)
        
        # 铁律2: 二阶导数 (极值 + 衰竭) -> Anti-Catch-Falling-Knife
        
        # --- 多头逻辑: 加息恐慌衰竭 (导致TLT暴跌的动能枯竭) ---
        # 1. 极端高位: 短端利率暴力拉升 (预期加息极值)
        cond_hawk_extreme = z_score > 1.2
        # 2. 动量衰竭: 拉升动能跌破3日均线 (恐慌见顶)
        cond_hawk_exhaust = velocity < velocity.rolling(3).mean()
        # 3. 交叉验证: 10Y-2Y曲线平坦化 (确认为短端驱动的熊平特征)
        cond_hawk_confirm = curve_change < 0
        
        bull_signal = cond_hawk_extreme & cond_hawk_exhaust & cond_hawk_confirm
        
        # --- 空头逻辑: 降息狂热衰竭 (导致TLT暴涨的动能枯竭) ---
        # 1. 极端低位: 短端利率暴力下杀 (预期降息极值)
        cond_dove_extreme = z_score < -1.2
        # 2. 动量衰竭: 下杀动能升破3日均线 (狂热见顶)
        cond_dove_exhaust = velocity > velocity.rolling(3).mean()
        # 3. 交叉验证: 10Y-2Y曲线陡峭化 (确认为短端驱动的牛陡特征)
        cond_dove_confirm = curve_change > 0
        
        bear_signal = cond_dove_extreme & cond_dove_exhaust & cond_dove_confirm
        
        # 赋值脉冲信号
        signal[bull_signal] = 1.0
        signal[bear_signal] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"