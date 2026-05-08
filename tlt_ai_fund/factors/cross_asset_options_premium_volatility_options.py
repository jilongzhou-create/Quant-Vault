import numpy as np
import pandas as pd

class CrossAssetOptionsPremiumFactor:
    """期权跨资产波动率溢价极值衰竭因子 (volatility/options)

    逻辑: 计算风险资产期权波动率(VIX)与避险资产期权波动率(GVZCLS)的差值作为恐慌溢价。当溢价极度狂飙时说明股票面临抛售，进入极度Risk-Off；当该极端溢价开始回落时(二阶导数衰竭)，标志着跨资产恐慌消退，流动性冲击瓦解，资金重返避险主核美债，触发做多脉冲。反之，当极度自满的负溢价反弹时，平静期被打破，触发做空脉冲。
    数据: vixcls, gvzcls
    触发: 波动率差值的126日 Z-Score > 1.5 且 差值 < 3日均值 触发多头脉冲；Z-Score < -1.5 且 差值 > 3日均值 触发空头脉冲。
    输出: +1.0 看多美债, -1.0 看空美债，非触发日常态输出 0.0。
    """

    def __init__(self, window: int = 126, z_thresh: float = 1.5, smooth: int = 3):
        self.name = 'cross_asset_options_premium'
        self.window = window
        self.z_thresh = z_thresh
        self.smooth = smooth

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 字段检查并前向填充处理缺失值
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 1. 计算期权波动率跨资产溢价 (量纲相同，均为年化百分比)
        spread = vix - gvz
        
        # 2. 计算滚动 Z-Score 以衡量水位的极端程度
        # 至少需要窗口一半的数据来计算均值和标准差
        roll_mean = spread.rolling(window=self.window, min_periods=self.window // 2).mean()
        roll_std = spread.rolling(window=self.window, min_periods=self.window // 2).std()
        
        # 防止除零异常
        z_score = (spread - roll_mean) / roll_std.replace(0, np.nan)
        
        # 3. 衰竭与边际变化条件 (二阶导数铁律)
        # 计算 3 日均值作为平滑基准，用于确认指标是否已见顶回落/见底回升
        spread_ma3 = spread.rolling(window=self.smooth).mean()
        
        # 4. 脉冲触发逻辑
        # 多头条件：股票相对黄金的恐慌溢价达到极值 (Z > 1.5) + 开始回落衰竭 (降至3日均值以下)
        # 经济学含义：非理性的流动性抛售结束，重归基本面，美债(TLT)迎来修复性上涨
        long_cond = (z_score > self.z_thresh) & (spread < spread_ma3)
        
        # 空头条件：极度自满或避险过度 (Z < -1.5) + 开始反弹发酵 (升至3日均值以上)
        # 经济学含义：极度平静被打破，风险事件重新定价，避险资金流出或债市遭遇通胀抛售
        short_cond = (z_score < -self.z_thresh) & (spread > spread_ma3)
        
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_thresh={self.z_thresh}, smooth={self.smooth})"