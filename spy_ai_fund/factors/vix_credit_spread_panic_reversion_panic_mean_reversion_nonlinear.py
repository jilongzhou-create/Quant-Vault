import numpy as np
import pandas as pd

class VixCreditSpreadPanicReversionFactor:
    """非线性特征交叉 (panic_mean_reversion/nonlinear)

    逻辑: 结合股市恐慌(VIX)和债市违约风险(High Yield OAS)。当极端恐慌(两者短期Z-Score双高)且VIX向下死叉3日均线时，恐慌情绪迎来衰竭，输出看多(抄底美股)；反之，若VIX温和上升(Z-score上穿0.5)且信用利差走阔，说明风险正在发酵，输出看空。严守不接飞刀的二阶导数法则。
    数据: [vixcls, bamlh0a0hym2]
    输出: 强看多(+1.0)表示恐慌见顶回落，强看空(-1.0)表示系统性风险萌芽。
    触发条件: 1. 做多: VIX Z(63)>1.2, OAS Z(63)>0.5, 且VIX今日跌破3日均线。2. 做空: VIX Z(63)刚上穿0.5且OAS上行。预期Trigger Rate 5%~10%。
    """

    def __init__(self):
        self.name = 'vix_credit_spread_panic_reversion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['vixcls', 'bamlh0a0hym2']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # 向前填充宏观日频数据的缺失值
        vix = data['vixcls'].ffill()
        oas = data['bamlh0a0hym2'].ffill()
        
        # 采用约一个季度(63个交易日)的窗口识别中期波段极值
        lookback = 63
        
        # 计算 Z-Scores (附加微小极值防止除0)
        vix_mean = vix.rolling(lookback).mean()
        vix_std = vix.rolling(lookback).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)
        
        oas_mean = oas.rolling(lookback).mean()
        oas_std = oas.rolling(lookback).std()
        oas_z = (oas - oas_mean) / (oas_std + 1e-8)
        
        vix_ma3 = vix.rolling(3).mean()
        
        # 衰竭脉冲 (Exhaustion Pulse): VIX 结束连续高涨，向下死叉3日均线的瞬间
        vix_exhaustion_pulse = (vix < vix_ma3) & (vix.shift(1) >= vix.shift(1).rolling(3).mean())
        
        # 风险发酵破位脉冲 (Breakout Pulse): VIX 从平静期苏醒，Z-score向上突破0.5
        vix_breakout_pulse = (vix_z > 0.5) & (vix_z.shift(1) <= 0.5)
        
        # --- 绝不接飞刀的多头抄底逻辑 ---
        # 只有在VIX高位(>1.2), OAS高位(>0.5), 且今天VIX动量明确转负(衰竭), OAS过去三天不再恶化时触发
        long_cond = (
            (vix_z > 1.2) &
            (oas_z > 0.5) &
            vix_exhaustion_pulse &
            (oas.diff(3) <= 0)
        )
        
        # --- 捕捉钝刀子割肉的空头逻辑 ---
        # 在VIX初现端倪，尚未到达主跌浪恐慌区，但信用债已经走阔时，发出预警卖空信号
        short_cond = (
            vix_breakout_pulse &
            (vix_z < 1.2) &
            (oas_z > 0.0) &
            (oas.diff(1) >= 0)
        )
        
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"