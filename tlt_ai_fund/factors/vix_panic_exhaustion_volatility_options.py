import numpy as np
import pandas as pd

class VixPanicExhaustionFactor:
    """VIX 恐慌瓦解反转因子 (volatility/options)

    逻辑: 当期权市场隐含波动率(VIX)处于极端高位或剧烈飙升时，往往伴随市场流动性挤兑和无差别抛售。随后当VIX脉冲破败并大幅回落时，标志着对冲盘解散与系统性恐慌消退，流动性恢复将推动被错杀的美债(TLT)价格强劲反弹。常态下处于休眠。
    数据: vixcls
    触发: VIX绝对水位或其短期动量的 252日 Z-Score > 1.5 (处于尾部恐慌极值)，且其日度跌幅超过 5% 并下穿 3日均线 (确认二阶导数衰竭) 时触发。
    输出: +1.0 脉冲信号，看多美债(TLT)。
    """

    def __init__(self):
        self.name = 'vix_panic_exhaustion_volatility_options'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns:
            return signal
            
        # 前向填充缺失值以防止数据间断
        vix = data['vixcls'].ffill()
        
        # 1. 绝对水位极值度量 (衡量年内极端恐慌状态，1.5 Z-score 对应约 93% 的偏右尾部分位数)
        vix_mean = vix.rolling(window=252).mean()
        vix_std = vix.rolling(window=252).std()
        vix_zscore = (vix - vix_mean) / vix_std
        
        # 2. 短期动量极值度量 (捕捉常态水位下，由突发事件引起的极陡峭飙升，捕捉斜率极值)
        vix_diff5 = vix.diff(5)
        vix_mom_mean = vix_diff5.rolling(window=252).mean()
        vix_mom_std = vix_diff5.rolling(window=252).std()
        vix_mom_zscore = (vix_diff5 - vix_mom_mean) / vix_mom_std
        
        # 处于以上任一恐慌极值状态，满足狙击手的“极端环境确认”条件
        extreme_state = (vix_zscore > 1.5) | (vix_mom_zscore > 1.5)
        
        # 3. 衰竭与边际变化条件 (防接飞刀，二阶导数铁律)
        vix_diff1 = vix.diff(1)
        vix_ma3 = vix.rolling(window=3).mean()
        pct_drop = vix.pct_change()
        
        # 要求波动率呈现明确的动量逆转：绝对变化回落、跌穿近3日均线均值、且单日跌幅超5%代表实质性平仓瓦解
        exhaustion_cond = (vix_diff1 < 0) & (vix < vix_ma3) & (pct_drop < -0.05)
        
        # 4. 触发多头脉冲信号
        trigger = extreme_state & exhaustion_cond
        signal[trigger] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"