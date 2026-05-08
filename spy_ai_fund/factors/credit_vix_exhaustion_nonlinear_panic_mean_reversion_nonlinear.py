import numpy as np
import pandas as pd

class CreditVixExhaustionNonlinearFactor:
    """信用利差与恐慌极值衰竭交叉因子 (panic_mean_reversion/nonlinear)

    逻辑: 捕捉美国高收益债信用利差与VIX共振飙升后的衰竭拐点，抄底长牛美股；而在恐慌未达极值且稳步恶化时看空。
    数据: vixcls (VIX指数), bamlh0a0hym2 (美国高收益债期权调整利差)
    输出: +1.0 表示极度恐慌见顶衰竭（强看多抄底），-1.0 表示轻微恐慌温水煮青蛙恶化（看空），0.0 为常态
    触发条件: VIX和利差的Z-score处于历史高位且今日双双回落触发多头；温和高位且近期稳步攀升触发空头。预期Trigger Rate 8%-12%。
    """

    def __init__(self):
        self.name = 'credit_vix_exhaustion_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['vixcls', 'bamlh0a0hym2']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # 填充缺失值，避免前视偏差
        vix = data['vixcls'].ffill()
        spread = data['bamlh0a0hym2'].ffill()
        
        # 计算 126个交易日(约半年) 的动态 Z-Score
        vix_mean = vix.rolling(window=126, min_periods=63).mean()
        vix_std = vix.rolling(window=126, min_periods=63).std().replace(0, 1e-5)
        vix_z = (vix - vix_mean) / vix_std
        
        spread_mean = spread.rolling(window=126, min_periods=63).mean()
        spread_std = spread.rolling(window=126, min_periods=63).std().replace(0, 1e-5)
        spread_z = (spread - spread_mean) / spread_std
        
        # 多头脉冲 (极端恐慌极值 + 今日开始衰竭回落)
        # 绝对禁止接飞刀，要求两者之任一处于极端恐慌水平，且动量和昨日边际变化双双由正转负
        extreme_panic = (vix_z > 1.5) | (spread_z > 1.5)
        panic_exhaustion = (vix.diff(1) < 0) & (spread.diff(1) < 0) & (vix < vix.rolling(3).mean())
        
        bull_condition = extreme_panic & panic_exhaustion
        
        # 空头脉冲 (微弱恐慌恶化，未达到出清极值)
        # Z-Score 在温和偏高范围(0.5 到 1.5 之间)，且利差和VIX连续攀升恶化
        creeping_panic = (vix_z > 0.5) & (vix_z <= 1.5) & (spread_z > 0.5) & (spread_z <= 1.5)
        trend_worsening = (vix.diff(3) > 1.0) & (spread.diff(3) > 0.1) & (vix.diff(1) > 0)
        
        bear_condition = creeping_panic & trend_worsening
        
        # 只在触发点输出脉冲信号
        signal[bear_condition] = -1.0
        signal[bull_condition] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"