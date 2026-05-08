import numpy as np
import pandas as pd

class OptionsVolatilityMicrostructureFactor:
    """期权微观结构去杠杆反转因子 (microstructure/options)

    逻辑: 风险平价(Risk Parity)和CTA基金严重依赖期权隐含波动率(VIX)进行机械性杠杆调节。当恐慌急剧飙升(VIX处于历史极值)时, 会引发微观结构上的流动性挤兑, 无差别抛售包含美债在内的资产。当恐慌极值开始衰竭回落时, 机器抛压解除, 美债等避险资产迎来脉冲式修复买盘; 反之, 极端自满后的突然飙升则预示抛压启动, 短期看空美债。
    数据: vixcls (标普500期权隐含波动率)
    触发: 
      - 看多: VIX 252日 Z-Score > 2.5 (恐慌极值) 且 VIX < 3日均值 (开始衰竭)
      - 看空: VIX 252日 Z-Score < -2.0 (由于VIX右偏分布,-2.0即为极度自满) 且 VIX > 3日均值 (反转飙升)
    输出: +1.0 (流动性抛压见顶反弹), -1.0 (自满破灭去杠杆抛压起), 其余时间严格为 0.0
    """

    def __init__(self, rolling_window: int = 252, long_z: float = 2.5, short_z: float = -2.0, decay_window: int = 3):
        self.name = 'options_volatility_microstructure_reversal'
        self.rolling_window = rolling_window
        self.long_z = long_z
        self.short_z = short_z
        self.decay_window = decay_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠, 初始信号严格设为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 数据缺失处理
        if 'vixcls' not in data.columns:
            return signal
            
        # 前向填充缺失值, 避免未来函数
        vix = data['vixcls'].ffill()
        
        # 铁律3: 边际变化, 不看绝对水位(如 >30), 而是看相对滚动窗口的统计偏离度
        roll_mean = vix.rolling(window=self.rolling_window).mean()
        roll_std = vix.rolling(window=self.rolling_window).std()
        
        # 避免除以零导致无穷大
        z_score = (vix - roll_mean) / roll_std.replace(0, np.nan)
        
        # 计算近期均值作为二阶导数反转基准
        vix_decay_mean = vix.rolling(window=self.decay_window).mean()
        
        # 铁律2: 二阶导数防接飞刀, 极值 + 衰竭
        # 多头触发: 流动性危机引发的无差别抛售结束, 抄底美债
        long_extreme = z_score > self.long_z
        long_exhaustion = vix < vix_decay_mean
        
        # 空头触发: 极度自满状态破灭, 机器开始降杠杆双抛股债, 做空避险资产对冲流动性冲击
        short_extreme = z_score < self.short_z
        short_exhaustion = vix > vix_decay_mean
        
        # 生成狙击手级别脉冲信号
        signal[long_extreme & long_exhaustion] = 1.0
        signal[short_extreme & short_exhaustion] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"OptionsVolatilityMicrostructureFactor(window={self.rolling_window}, long_z={self.long_z}, short_z={self.short_z}, decay={self.decay_window})"