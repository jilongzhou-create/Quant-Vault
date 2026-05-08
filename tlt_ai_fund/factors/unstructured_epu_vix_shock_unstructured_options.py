import numpy as np
import pandas as pd

class UnstructuredFomcOptionsExhaustionFactor:
    """Unstructured Fomc Options Exhaustion Factor (unstructured/options)

    逻辑: 结合期权隐含波动率(VIX/GVZ)的极端衰竭与非结构化FOMC情绪的边际突变。常态下波动率脉冲衰竭代表纯粹的流动性危机解除或避险情绪启动，利好美债(+1.0)；但在FOMC鹰派情绪突变(加息恐慌)的宏观状态下，美债成为抛售源头，任何波动率脉冲及其衰竭均指示逢高做空(-1.0)，完美解决条件IC为负的均值回归陷阱。
    数据: vixcls (Options), gvzcls (Options), fomc_sentiment (Unstructured).
    触发: VIX/跨资产波动率差值 Z-Score > 2.5 且开始回落 (二阶导数衰竭) + FOMC情绪 5日边际变化的 Z-Score 识别宏观鹰派突变。
    输出: 狙击手级脉冲信号 [-1.0, 1.0]。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_options_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 必须验证所需领域数据是否存在
        if 'vixcls' not in data.columns or 'fomc_sentiment' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        fomc = data['fomc_sentiment'].ffill()
        
        # 获取可选的GVZ数据(跨资产黄金波动率)
        if 'gvzcls' in data.columns:
            gvz = data['gvzcls'].ffill()
        else:
            gvz = pd.Series(np.nan, index=data.index)
            
        vol_window = 63  # 短窗口确保能捕捉局部极端波动脉冲
        
        # ---------------------------------------------------------
        # 1. Options 波动率微观结构: 极端极值与衰竭探测 (狙击手扳机)
        # ---------------------------------------------------------
        vix_mean = vix.rolling(vol_window).mean()
        vix_std = vix.rolling(vol_window).std().replace(0, np.nan)
        
        # 绝对高位(恐慌)与绝对低位(极度自满)的Z-Score
        vix_z_up = (vix - vix_mean) / vix_std
        vix_z_down = (vix_mean - vix) / vix_std
        vix_ma3 = vix.rolling(3).mean()
        
        # 铁律2: 二阶导数探测 -> 恐慌衰竭 (极高 + 开始回落)
        panic_pulse = (vix_z_up > 2.5) & (vix < vix_ma3) & (vix.diff() < 0)
        
        # 铁律2: 二阶导数探测 -> 自满衰竭 (极低 + 开始苏醒)
        complacency_pulse = (vix_z_down > 2.5) & (vix > vix_ma3) & (vix.diff() > 0)
        
        # 跨资产恐慌(股市VIX 相对 金市GVZ 超涨)的衰竭
        spread_pulse = pd.Series(False, index=data.index)
        if not gvz.isna().all():
            spread = vix - gvz
            spread_mean = spread.rolling(vol_window).mean()
            spread_std = spread.rolling(vol_window).std().replace(0, np.nan)
            spread_z_up = (spread - spread_mean) / spread_std
            spread_ma3 = spread.rolling(3).mean()
            
            spread_pulse = (spread_z_up > 2.5) & (spread < spread_ma3) & (spread.diff() < 0)
            
        # 只要满足任何一种期权波动率极值衰竭即视为触发脉冲
        any_pulse = panic_pulse | complacency_pulse | spread_pulse
        
        # ---------------------------------------------------------
        # 2. Unstructured FOMC情绪: 边际突变状态识别 (因果方向滤波器)
        # ---------------------------------------------------------
        # 铁律3: 低频阶梯状数据绝对禁止直接用绝对值，必须提取边际变化(diff)
        fomc_window = 126
        fomc_diff = fomc.diff(5)
        fomc_diff_mean = fomc_diff.rolling(fomc_window).mean()
        fomc_diff_std = fomc_diff.rolling(fomc_window).std().replace(0, np.nan)
        
        fomc_z = (fomc_diff - fomc_diff_mean) / fomc_diff_std
        
        # 识别鹰派突变(FOMC得分急降，Z-Score < -2.0)
        # 将鹰派冲击的记忆保留 42 个交易日 (覆盖一整个加息发酵周期)
        hawkish_shock = (fomc_z < -2.0).rolling(42).max() > 0
        hawkish_shock = hawkish_shock.fillna(False)
        
        # ---------------------------------------------------------
        # 3. 信号生成与赋值
        # ---------------------------------------------------------
        # 默认基础逻辑: 波动率脉冲衰竭代表避险资金归位或流动性冲击解除，利好美债 (+1.0)
        signal[any_pulse] = 1.0
        
        # 修正 IC 为负的核心逻辑: 若当前处于鹰派突变发酵期，债券本身就是被抛售的风险源。
        # 此时任何波动率的爆发和衰竭都意味着利率中枢的上移，逢高果断做空美债 (-1.0)
        signal[any_pulse & hawkish_shock] = -1.0
        
        # 清理缺失值，保证在没有触发事件的交易日绝对静默 (0.0)
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"