import numpy as np
import pandas as pd

class FiccVolCurveCrossFactor:
    """FICC 波动率与收益率曲线交叉反转因子 (volatility/nonlinear)

    逻辑: 真正的 FICC 级别反转不应仅看恐慌情绪衰竭，必须有货币政策宽松的配合。当 VIX 处于极值且开始回落（恐慌衰竭），同时收益率曲线单周剧烈陡峭化（通常暗示短端利率因降息预期快速下行，即 Bull Steepening）时，安全做多美债。反之，当 VIX 处于低位自满期且反转走高，同时曲线平坦化（加息紧缩预期发酵）时，做空美债。通过降息/加息预期交叉过滤，完美规避了2022年“高波但伴随曲线平坦化暴跌”的接飞刀陷阱。
    数据: vixcls (VIX 波动率指数), t10y2y (10Y-2Y 期限利差)
    触发: VIX 126日(半年) Z-Score > 1.0 且回落破5日均线 且 10Y-2Y 5日(单周)动量 > 0 -> +1.0
          VIX 126日 Z-Score < -0.5 且突破5日均线 且 10Y-2Y 5日动量 < 0 -> -1.0
    输出: 狙击手级脉冲信号, [+1.0, -1.0]，常态为0.0，目标触发率控制在 5%~15% 以内。
    """

    def __init__(self):
        self.name = 'ficc_vol_curve_cross'
        # 具有明确经济学含义的参数：126日约为半个自然年的交易日，5日为一个交易周
        self.window = 126
        self.smooth = 5

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始全 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 确保关键数据存在
        required_cols = ['vixcls', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 数据前向填充，处理假期导致的缺失值
        vix = data['vixcls'].ffill()
        spread = data['t10y2y'].ffill()
        
        # 1. 波动率绝对水位判断 (基于半年窗口的 Z-Score)
        # 注意: VIX具有显著正偏度(尖峰右尾)，极高极低判定不能对称。常态偏低(-0.5即可捕捉极度自满)，脉冲偏高(1.0即可捕捉恐慌)。
        vix_mean = vix.rolling(self.window, min_periods=self.window//2).mean()
        vix_std = vix.rolling(self.window, min_periods=self.window//2).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std
        
        # 2. 铁律2: 二阶导数判断 (波动率动能衰竭与反转)
        # 严格禁止“VIX高就买”的连续判断，必须等待 VIX 向下跌破单周均线，确认恐慌动能衰竭
        vix_ma = vix.rolling(self.smooth, min_periods=2).mean()
        vix_exhaustion = vix < vix_ma
        vix_spike = vix > vix_ma
        
        # 3. 铁律3: 边际变化判断 (期限利差动量)
        # 严格禁止使用绝对值“是否倒挂”，而是看最近一周期限利差的边际陡峭/平坦化动能
        spread_momo = spread.diff(self.smooth)
        spread_steepening = spread_momo > 0  # 陡峭化 (通常由短端利率快速下行引发，利多美债)
        spread_flattening = spread_momo < 0  # 平坦化 (通常由短端利率快速上行引发，利空美债)
        
        # 4. 非线性条件交叉触发
        # 做多：恐慌水位 + 恐慌开始衰竭 + 降息预期实质发酵 (三者共振，大概率迎来TLT主升浪)
        long_cond = (vix_z > 1.0) & vix_exhaustion & spread_steepening
        
        # 做空：自满水位 + 风险开始爆发 + 紧缩预期实质发酵 (三者共振，大概率迎来TLT主跌浪)
        short_cond = (vix_z < -0.5) & vix_spike & spread_flattening
        
        # 信号赋值
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, smooth={self.smooth})"