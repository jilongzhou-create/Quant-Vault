import numpy as np
import pandas as pd

class OptionVolCrowdingReversalFactor:
    """期权波动率极端拥挤与反转因子 (volatility/options)

    逻辑: 监控期权市场波动率(VIX)及其变动率(VVIX代理)的极端狂飙现象。当对冲盘极度拥挤导致波动率达到极端高位(Z-Score>2.5)并开始瓦解回落时，表明做市商Gamma挤压结束、恐慌消退，资金重新回流美债锁定避险高息，此时触发看多(脉冲)。反之，当市场极度自满(Z-Score<-2.0)且风险重新抬头时，平抑波动率的杠杆套利盘解体，通常引发股债双杀，此时触发看空(脉冲)。
    数据: vixcls
    触发: VIX 或 VVIX 252日Z-Score > 2.5 且 diff() < 0 且跌破3日均线 -> +1.0；VIX Z-Score < -2.0 且 diff() > 0 且突破3日均线 -> -1.0
    输出: +1.0(看多TLT), -1.0(看空TLT), 其余为0.0 (狙击手级脉冲信号)
    """

    def __init__(self):
        self.name = 'option_vol_crowding_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        # 必须处理缺失该字段的情况
        if 'vixcls' not in data.columns:
            return signal
            
        # 获取有效数据并前向填充
        vix = data['vixcls'].ffill()
        
        # 1. 波动率的 Z-Score (边际变化水位评估)
        vix_mean = vix.rolling(window=252, min_periods=126).mean()
        vix_std = vix.rolling(window=252, min_periods=126).std()
        vix_z = (vix - vix_mean) / vix_std
        
        # 2. 波动率变动动能均线 (微观结构：短期平滑捕捉衰竭转折点)
        vix_ma3 = vix.rolling(window=3).mean()
        
        # 3. 构造 VVIX 代理 (Volatility of Volatility)
        # 使用对数收益率的 20 日滚动标准差来代理期权市场的加速度风险
        vix_log_ret = np.log(vix / vix.shift(1).replace(0, np.nan))
        vvix_proxy = vix_log_ret.rolling(window=20).std()
        vvix_mean = vvix_proxy.rolling(window=252, min_periods=126).mean()
        vvix_std = vvix_proxy.rolling(window=252, min_periods=126).std()
        vvix_z = (vvix_proxy - vvix_mean) / vvix_std
        
        # --- 核心二阶导数铁律: 极值 + 衰竭 ---
        
        # 多头脉冲: 极端恐慌 + 开始回落
        # 极值条件: VIX 或 VVIX 达到极其罕见的高位
        long_extreme = (vix_z > 2.5) | (vvix_z > 2.5)
        # 衰竭条件: 绝对变化小于0，且跌破3日短期均线
        long_exhaustion = (vix.diff() < 0) & (vix < vix_ma3)
        long_cond = long_extreme & long_exhaustion
        
        # 空头脉冲: 极端自满 + 风险抬头
        # 极值条件: VIX 处于长期的不正常低位
        short_extreme = (vix_z < -2.0)
        # 衰竭(反向抬头)条件: 绝对变化大于0，且突破3日短期均线
        short_exhaustion = (vix.diff() > 0) & (vix > vix_ma3)
        short_cond = short_extreme & short_exhaustion
        
        # 信号赋值，输出脉冲
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        # 补充清理缺失值产生的杂音
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"