import numpy as np
import pandas as pd

class CrossAssetImpliedVolExhaustionFactor:
    """期权隐含波动率极值衰竭因子 (volatility/options)

    逻辑: 捕捉股(VIX)金(GVZ)跨资产期权隐含波动率的局部极度狂飙与拥挤瓦解。由于期权买盘具有强烈的主跌浪"接飞刀"特征，只在波动率处于极端高位(Z-Score>2.5)且恐慌情绪开始边际消退(下破3日均线)时，才确认系统性流动性抛售结束并输出看多美债脉冲；反之，在极端自满期波动率初现反弹端倪时，输出看空脉冲。该因子常态严格休眠，仅在拐点瞬间狙击。
    数据: vixcls (CBOE VIX波动率), gvzcls (CBOE 黄金ETF隐含波动率)
    触发: 极度恐慌(Z-Score > 2.5) + 同步衰竭(<3日均值) 触发看多 +1.0；极度自满(Z-Score < -2.0) + 同步抬头(>3日均值) 触发看空 -1.0。
    输出: [-1.0, 1.0] 狙击手级脉冲信号。
    """

    def __init__(self):
        self.name = 'cross_asset_implied_vol_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必需的期权波动率数据字段
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 安全填充与数据对齐，避免零或负值引发 log 计算警告
        vix = data['vixcls'].ffill().replace(0, np.nan)
        gvz = data['gvzcls'].ffill().replace(0, np.nan)
        
        vix = vix.reindex(data.index).ffill()
        gvz = gvz.reindex(data.index).ffill()

        # 转换为对数序列平抑偏度，增强隐含波动率 Z-Score 的正态分布统计意义
        log_vix = np.log(vix)
        log_gvz = np.log(gvz)

        # 使用短周期(21个交易日，约1个月)计算动态 Z-Score，敏锐捕捉期权对冲盘的短期极度拥挤
        window = 21
        
        # VIX Z-Score 与 动量变化(二阶导)
        vix_mean = log_vix.rolling(window).mean()
        vix_std = log_vix.rolling(window).std()
        vix_z = (log_vix - vix_mean) / vix_std
        
        vix_ma3 = vix.rolling(3).mean()
        vix_exhausted = vix < vix_ma3
        vix_surging = vix > vix_ma3

        # GVZ Z-Score 与 动量变化(二阶导)
        gvz_mean = log_gvz.rolling(window).mean()
        gvz_std = log_gvz.rolling(window).std()
        gvz_z = (log_gvz - gvz_mean) / gvz_std
        
        gvz_ma3 = gvz.rolling(3).mean()
        gvz_exhausted = gvz < gvz_ma3
        gvz_surging = gvz > gvz_ma3

        # 核心逻辑：极端恐慌（任一资产期权波动率异动飙升）
        extreme_panic = (vix_z > 2.5) | (gvz_z > 2.5)
        # 衰竭确认：跨资产期权波动率必须同步确认回落（杜绝接飞刀）
        panic_exhaustion = vix_exhausted & gvz_exhausted

        # 核心逻辑：极端自满（期权做空对冲极度拥挤）
        extreme_complacency = (vix_z < -2.0) | (gvz_z < -2.0)
        # 抬升确认：跨资产期权波动率同步掉头向上（自满瓦解前夕）
        complacency_reversal = vix_surging & gvz_surging

        # 组合判定信号
        long_pulse = extreme_panic & panic_exhaustion
        short_pulse = extreme_complacency & complacency_reversal

        # 注入信号：非触发日严格保持 0.0 (Sniper Pulse)
        signal[long_pulse] = 1.0
        signal[short_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"