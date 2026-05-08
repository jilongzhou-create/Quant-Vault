import numpy as np
import pandas as pd

class MacroPanicYieldCurveExhaustionFactor:
    """宏观恐慌与期限利差衰竭因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合VIX与期限利差(T10Y2Y)的非线性特征。当市场极度恐慌(VIX Z-score > 1.5)且期限利差急剧变陡(危机爆发倒挂解除的标志)时，若恐慌开始衰竭(VIX回落)，代表流动性挤兑极点已过，产生强烈看多抄底脉冲；而在轻度恐慌爬升且曲线持续平坦化(紧缩预期恶化)时，代表钝刀割肉的主跌浪前奏，产生看空脉冲。常态下零值休眠。
    数据: vixcls (波动率), t10y2y (10年期与2年期国债利差)
    输出: 强看多(+1.0) / 看空(-1.0) 脉冲信号，常态返回 0.0
    触发条件: 多头脉冲要求VIX极高+回落且曲线近期变陡；空头要求VIX温和上升且曲线平坦化。目标 Trigger Rate 在 6%-12% 之间。
    """

    def __init__(self):
        self.name = 'macro_panic_yield_curve_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 默认信号输出 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 必须数据列存在
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        # 前向填充数据，避免NaN导致的计算中断
        df = pd.DataFrame({
            'vix': data['vixcls'],
            't10y2y': data['t10y2y']
        }).ffill()
        
        if df.dropna().empty:
            return signal
            
        # VIX 的 126 日(半年)滚动 Z-Score 计算，用于衡量极值状态
        vix_mean = df['vix'].rolling(window=126, min_periods=60).mean()
        vix_std = df['vix'].rolling(window=126, min_periods=60).std()
        vix_z = (df['vix'] - vix_mean) / (vix_std + 1e-6)
        
        # === 1. 多头抄底逻辑 (极端恐慌极值 + 二阶导衰竭) ===
        # 恐慌开始回落：当日下跌 且 低于过去3日均值
        vix_exhaustion = (df['vix'].diff(1) < 0) & (df['vix'] < df['vix'].rolling(window=3).mean())
        # 宏观验证：过去20个交易日内期限利差走阔/陡峭化 > 5个基点 (危机应对降息预期)
        curve_steepening = df['t10y2y'].diff(20) > 0.05
        
        # 触发条件聚合
        long_cond = (vix_z > 1.5) & vix_exhaustion & curve_steepening
        
        # === 2. 空头避险逻辑 (轻度恐慌恶化 + 流动性紧缩预期) ===
        # VIX处于温水煮青蛙的爬升期
        vix_rising = (df['vix'].diff(1) > 0) & (df['vix'] > df['vix'].rolling(window=5).mean())
        # 宏观验证：过去10个交易日内期限利差平坦化/倒挂加深 < -5个基点 (经济紧缩定价加强)
        curve_flattening = df['t10y2y'].diff(10) < -0.05
        
        # 触发条件聚合: Z-score不在极端高位(尚未见底)，但处于加速上行期
        short_cond = (vix_z > 0.5) & (vix_z <= 1.5) & vix_rising & curve_flattening
        
        # 将条件映射到源索引
        long_idx = long_cond.reindex(signal.index).fillna(False)
        short_idx = short_cond.reindex(signal.index).fillna(False)
        
        signal.loc[long_idx] = 1.0
        signal.loc[short_idx] = -1.0
        
        # 冲突防御性归0 (理论上不可能同时满足)
        conflict = long_idx & short_idx
        signal.loc[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"