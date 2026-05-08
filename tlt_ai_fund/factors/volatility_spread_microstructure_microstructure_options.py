import numpy as np
import pandas as pd

class MicrostructureOptionsVolCurveFactor:
    """Volatility Regime and Yield Curve Microstructure Factor

    逻辑: 采用非线性波动率微观结构捕捉美债的多空脉冲。在常态恐慌(VIX Z在1.2~2.5)见顶衰竭时，避险资金流出导致美债下跌(-1.0)；但在极端流动性危机(Z>2.5)见顶衰竭时，联储救市带来美债暴涨(+1.0)。同时捕捉VIX自满破裂和收益率曲线平坦极值后的突发牛陡(Bull Steepening)，输出看多美债脉冲。
    数据: vixcls, t10y2y
    触发: VIX 63日Z-Score结合2日diff动量，T10Y2Y的Z-Score结合动量陡峭化。
    输出: [-1.0, 1.0] 的极短期脉冲信号。
    """

    def __init__(self):
        self.name = 'microstructure_options_vol_curve'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 基础数据校验与提取
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        # 前向填充缺失值，避免未来数据泄漏
        vix = data['vixcls'].ffill()
        curve = data['t10y2y'].ffill()
        
        # 2. 计算微观结构极值 (使用63个交易日即约1个季度的滚动窗口捕捉中短期Regime)
        vix_z = (vix - vix.rolling(63).mean()) / vix.rolling(63).std()
        curve_z = (curve - curve.rolling(63).mean()) / curve.rolling(63).std()
        
        # 3. 边际变化：计算2日动量，严格遵守二阶导数与边际变化铁律
        vix_diff = vix.diff(2)
        curve_diff = curve.diff(2)
        
        # 初始信号必须全为 0.0 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 4. 触发逻辑定义 (非线性条件划分)
        
        # [看空 -1.0] 常态恐慌见顶衰竭：VIX处于高位但开始回落，风险偏好回归(Risk-On)，股市上涨避险债市下跌。
        normal_panic_exhaustion = (vix_z > 1.2) & (vix_z <= 2.5) & (vix_diff < -1.0)
        
        # [看多 +1.0] 极端流动性危机衰竭：极值恐慌回落意味着联储干预(QE)，无差别抛售(Dash for Cash)结束，美债报复性反弹。
        liquidity_crisis_exhaustion = (vix_z > 2.5) & (vix_diff < -1.0)
        
        # [看多 +1.0] 自满情绪破裂：VIX极低时突然飙升，典型的突发Risk-Off事件，避险资金涌入美债。
        complacency_breakout = (vix_z < -0.8) & (vix_diff > 1.5)
        
        # [看多 +1.0] 收益率曲线突发牛陡 (Bull Steepening)：曲线极度平坦或倒挂后突发陡峭，强烈暗示联储紧急降息预期。
        bull_steepening = (curve_z < -1.0) & (curve_diff > 0.04)
        
        # 5. 信号赋值 (按照优先级顺序，后赋值的覆盖前面的，确保极端事件优先)
        signal[normal_panic_exhaustion] = -1.0
        signal[complacency_breakout] = 1.0
        signal[bull_steepening] = 1.0
        signal[liquidity_crisis_exhaustion] = 1.0
        
        # 处理可能的 NaN (滚动窗口初始阶段) 并命名
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"