import numpy as np
import pandas as pd

class VolCurveSteepeningNexusFactor:
    """波动率衰竭与曲线陡峭共振因子 (Volatility/Nonlinear)

    逻辑: 纯 VIX 的均值回归容易在主跌浪中接飞刀。真正的宏观底部脉冲发生于两大引擎共振: 当股市恐慌(VIX)在季度级别触及极端高位并开始实质性回落(二阶导数衰竭)时, 且 FICC 内部的收益率曲线(10Y-2Y)出现剧烈的边际变陡(短端利率崩盘, 市场急速为美联储降息/注入流动性定价)。两个条件缺一不可。反之, 在 VIX 极度低迷(波动率做空拥挤)且被突然打破时, 若曲线剧烈平坦化(短端利率狂飙, 紧缩冲击), 则输出看空脉冲。
    数据: vixcls (VIX指数), t10y2y (10年-2年期限利差)
    触发: 做多: VIX 63日 Z-Score > 2.0 且低于3日均线(衰竭) + 利差5日动量 > 10bps(剧烈陡峭化)。做空: VIX Z-Score < -1.5 且突破3日均线 + 利差5日动量 < -10bps(剧烈平坦化)。
    输出: +1.0 (恐慌衰竭与降息定价共振看多), -1.0 (拥挤瓦解与紧缩冲击看空), 常态绝对休眠为 0.0。
    """

    def __init__(self):
        self.name = 'vol_curve_steepening_nexus'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠初始化
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            return signal

        # 前向填充处理交易日错位
        vix = data['vixcls'].ffill()
        curve = data['t10y2y'].ffill()

        # 计算 VIX 季度级(63个交易日) Z-Score 识别极值
        # 阈值 2.0 和 -1.5 具有统计学意义, 筛选尾部 2% ~ 6% 的极端情绪分布
        vix_mean_63 = vix.rolling(window=63).mean()
        vix_std_63 = vix.rolling(window=63).std().replace(0, np.nan)
        vix_z = (vix - vix_mean_63) / vix_std_63

        # 铁律2: 二阶导数 (防飞刀) - 必须出现转折
        vix_ma3 = vix.rolling(window=3).mean()
        vix_exhaustion = vix < vix_ma3  # 恐慌开始消退
        vix_wakeup = vix > vix_ma3      # 极度平静被打破

        # 铁律3: 边际变化 - 绝对不看曲线是否倒挂, 只看瞬间的动量爆发(Bull Steepening / Bear Flattening)
        # 5日内利差变化超过 0.10 (即 10个基点), 代表宏观预期的剧烈重新定价
        curve_mom_5d = curve.diff(5)
        curve_steepening = curve_mom_5d > 0.10
        curve_flattening = curve_mom_5d < -0.10

        # --- 信号触发逻辑 ---
        
        # 多头触发: 恐慌极值 + 恐慌回落衰竭 + 收益率曲线因放水预期剧烈陡峭化
        long_trigger = (vix_z > 2.0) & vix_exhaustion & curve_steepening
        
        # 空头触发: 极度拥挤的低波动 + 波动率突然飙升 + 收益率曲线因紧缩预期剧烈平坦化
        short_trigger = (vix_z < -1.5) & vix_wakeup & curve_flattening

        # 赋值狙击手脉冲信号
        signal[long_trigger] = 1.0
        signal[short_trigger] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"