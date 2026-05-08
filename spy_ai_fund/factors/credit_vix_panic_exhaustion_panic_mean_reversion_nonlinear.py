import numpy as np
import pandas as pd

class CreditVixPanicExhaustionFactor:
    """恐慌极值与均值回归 (panic_mean_reversion/nonlinear)

    逻辑: 信用利差与股市波动率是非线性互补的风险溢价指标。当联合极度恐慌且今日双双回落时，标志危机缓解和抄底良机(均值回归)；反之，常态下二者若突然同时急剧走阔，则预示趋势恶化。
    数据: [vixcls, bamlh0a0hym2]
    输出: +1.0 表示恐慌极值衰竭（抄底买点），-1.0 表示常态下风险突增（短线看空），0.0 为常态休眠。
    触发条件: VIX或OAS的252日Z-Score>1.5且双边日度边际下降时触发+1.0；Z-Score<1.0且5日内急剧走阔(动量突变)时触发单日-1.0脉冲。预期Trigger Rate 6%-10%。
    """

    def __init__(self):
        self.name = 'credit_vix_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['vixcls', 'bamlh0a0hym2']
        if not all(col in data.columns for col in req_cols):
            return signal
            
        # 填充缺失值并提取数据
        vix = data['vixcls'].ffill()
        oas = data['bamlh0a0hym2'].ffill()
        
        # 计算 252 日滚动的 Z-score，加极小值防除零
        vix_z = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-6)
        oas_z = (oas - oas.rolling(252).mean()) / (oas.rolling(252).std() + 1e-6)
        
        # 防止初始NaN值导致比较出错
        vix_z = vix_z.fillna(0.0)
        oas_z = oas_z.fillna(0.0)
        
        # ==========================================
        # 抄底条件 (+1.0)：极度恐慌 + 衰竭
        # ==========================================
        # 联合极值判断：至少有一个维度处于 1.5个标准差以上的极端恐慌水平
        is_panic = (vix_z > 1.5) | (oas_z > 1.5)
        
        # 二阶导数铁律：必须等待恐慌情绪见顶回落，防接飞刀
        # VIX今日下降，且高收益债OAS没有继续恶化
        is_exhausting = (vix.diff(1) < 0) & (oas.diff(1) <= 0)
        
        buy_pulse = is_panic & is_exhausting
        
        # ==========================================
        # 恶化条件 (-1.0)：常态下的恐慌急剧爆发
        # ==========================================
        # 避开极值状态做空，防止在历史性大底杀跌卖出
        not_panic = (vix_z < 1.0) & (oas_z < 1.0)
        
        # 动量激增：VIX 5天内飙升超过 3.0，同时信用利差急剧走阔 > 30 bps (0.30)
        vix_soaring = vix.diff(5) > 3.0
        oas_soaring = oas.diff(5) > 0.30
        
        bear_state = not_panic & vix_soaring & oas_soaring
        # 边缘跳变铁律：为了保证狙击手级别的脉冲信号，只在刚好满足爆发条件的瞬间输出一次
        bear_pulse = bear_state & ~bear_state.shift(1).fillna(False)
        
        # ==========================================
        # 信号赋值
        # ==========================================
        signal[bear_pulse] = -1.0
        signal[buy_pulse] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"