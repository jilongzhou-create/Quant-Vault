import numpy as np
import pandas as pd

class OptionsGammaSqueezeReversalFactor:
    """期权 Gamma 挤压反转因子 (volatility/options)

    逻辑: 捕捉期权市场极端 Gamma Squeeze（做市商被迫买入或抛售导致隐含波动率单日产生极端跳跃）的耗尽反转时刻。
          当 VIX 发生极端脉冲暴涨后并开始边际回落时，标志着对冲盘踩踏平仓的结束，流动性危机解除，此时避险与宽货币预期共同推动美债价格强烈脉冲式反弹。
          这种微观结构上的恐慌衰竭极度短暂，常态下必须保持零值休眠，仅在破局窗口输出信号。
    数据: vixcls (CBOE VIX 隐含波动率)
    触发: VIX 单日变动的 126日 Z-Score > 2.5 (极致恐慌脉冲) 且在 3 日内出现二阶衰竭 (单日变动 < 0 且跌穿 3日均线) -> +1.0
          VIX 单日变动的 126日 Z-Score < -2.5 (极致自满脉冲) 且在 3 日内出现二阶衰竭 (单日变动 > 0 且突破 3日均线) -> -1.0
    输出: +1.0 看多美债(恐慌耗尽), -1.0 看空美债(极度自满解除), 0.0 常态休眠
    """

    def __init__(self):
        self.name = 'options_gamma_squeeze_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化常态休眠信号为 0.0
        signal = pd.Series(0.0, index=data.index)

        # 数据校验保护
        if 'vixcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        
        # ---------------- 铁律 3: 边际变化 (捕获恐慌瞬间加速度) ---------------- #
        vix_diff = vix.diff()
        
        # 计算变动量的 126日 (半年) 动态 Z-Score，反映相对于当前宏观环境的“惊恐度”
        vix_diff_mean = vix_diff.rolling(window=126, min_periods=30).mean()
        vix_diff_std = vix_diff.rolling(window=126, min_periods=30).std()
        vix_diff_z = (vix_diff - vix_diff_mean) / (vix_diff_std + 1e-8)
        
        # ---------------- 铁律 1: 零值休眠与狙击点 (极端条件监控) ---------------- #
        # 记录近 3 日内是否发生过极端恐慌暴涨或自满暴跌 (Z-Score > 2.5 或 < -2.5)
        extreme_spike = vix_diff_z > 2.5
        recent_spike = extreme_spike.rolling(window=3, min_periods=1).max() > 0
        
        extreme_plunge = vix_diff_z < -2.5
        recent_plunge = extreme_plunge.rolling(window=3, min_periods=1).max() > 0
        
        # 辅助安全垫：确保反转时的绝对水位具备意义 (防止低波泥潭里的无效微动)
        vix_level_mean = vix.rolling(window=126, min_periods=30).mean()
        vix_level_std = vix.rolling(window=126, min_periods=30).std()
        vix_level_z = (vix - vix_level_mean) / (vix_level_std + 1e-8)
        
        # ---------------- 铁律 2: 二阶导数 (绝不接飞刀，必须等动能衰竭) ---------------- #
        vix_ma3 = vix.rolling(window=3, min_periods=1).mean()
        
        # 多头衰竭: 曾经极度恐慌，但在今天 VIX 停止上涨，且跌破短期均线
        exhaustion_up = (vix_diff < 0) & (vix < vix_ma3)
        
        # 空头衰竭: 曾经极度自满，但在今天 VIX 停止下跌，且突破短期均线
        exhaustion_down = (vix_diff > 0) & (vix > vix_ma3)
        
        # 生成狙击手级脉冲信号
        cond_long = recent_spike & exhaustion_up & (vix_level_z > 0.5)
        cond_short = recent_plunge & exhaustion_down & (vix_level_z < -0.5)
        
        signal.loc[cond_long] = 1.0
        signal.loc[cond_short] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"