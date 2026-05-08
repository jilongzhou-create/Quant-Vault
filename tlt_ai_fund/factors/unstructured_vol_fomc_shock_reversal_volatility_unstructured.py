import numpy as np
import pandas as pd

class YieldCurveUncertaintyReversalFactor:
    """Yield Curve Panic & Uncertainty Exhaustion (volatility/unstructured)

    逻辑: 捕捉收益率曲线极值脉冲与政策不确定性衰竭的跨维共振。绝对避免了纯期权隐含波动率(VIX)的拥挤因子。当加息恐慌引发极端熊平(Bear Flattening, 曲线动量极度向下, Z<-2.5), 且该恐慌开始衰竭、非结构化政策不确定性(EPU)同步回落时, 加息预期见顶, 脉冲做多美债。当衰退恐慌引发极端牛陡(Bull Steepening, 曲线动量飙升, Z>2.5), 且陡峭化动量衰竭、EPU回落时, 避险拥挤瓦解, 脉冲做空美债。
    数据: t10y2y (收益率曲线利差, 代表FICC核心定价), usepuindxd (经济政策不确定性指数, NLP非结构化数据)
    触发: t10y2y 3日变化量 252日 Z-Score 突破 ±2.5 (极值), 且曲线近期动量反转 (二阶衰竭), 且 EPU 确认降温。
    输出: +1.0 (恐慌见顶反转看多) / -1.0 (避险见顶反转看空), 狙击手脉冲信号保持极短几天。
    """

    def __init__(self):
        self.name = 'yc_epu_panic_reversal_unstructured'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理缺少所需列的情况，直接返回全 0 Series
        required_cols = ['t10y2y', 'usepuindxd']
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        # 提取并前向填充缺失值, 防止由于节假日错位产生的 NaN
        t10y2y = data['t10y2y'].ffill()
        epu = data['usepuindxd'].ffill()
        
        # 核心铁律3: 边际变化 (Marginal Change Only)
        # 计算收益率曲线的短期动量 (Steepening/Flattening Velocity)
        yc_vel = t10y2y.diff(3)
        
        # 计算曲线动量的 252日 Z-Score, 衡量波动率极端水平
        roll_mean = yc_vel.rolling(window=252, min_periods=63).mean()
        roll_std = yc_vel.rolling(window=252, min_periods=63).std().replace(0, np.nan)
        yc_z = (yc_vel - roll_mean) / roll_std
        
        # 核心铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # Bear Flattening (极端变平) 的衰竭: 曲线停止下跌, 2日动量转正
        flattening_exhausted = t10y2y.diff(2) > 0
        
        # Bull Steepening (极端变陡) 的衰竭: 曲线停止飙升, 2日动量转负
        steepening_exhausted = t10y2y.diff(2) < 0
        
        # 跨维确认: 非结构化数据 EPU 政策不确定性开始降温 (同样是边际变化)
        epu_cooling = epu.diff(3) < 0
        
        # 核心铁律1: 零值休眠 (Sniper Pulse)
        raw_signal = pd.Series(0.0, index=data.index)
        
        # 信号逻辑 A: 极端加息恐慌引发的暴跌见底 -> 做多美债 (+1.0)
        # 极值(Z < -2.5) + 衰竭(曲线动量反转) + EPU降温
        long_cond = (yc_z < -2.5) & flattening_exhausted & epu_cooling
        
        # 信号逻辑 B: 极端衰退恐慌引发的暴涨见顶 -> 做空美债 (-1.0)
        # 极值(Z > 2.5) + 衰竭(曲线动量反转) + EPU降温
        short_cond = (yc_z > 2.5) & steepening_exhausted & epu_cooling
        
        raw_signal.loc[long_cond] = 1.0
        raw_signal.loc[short_cond] = -1.0
        
        # 将脉冲信号保持 4 个交易日 (加上触发日共 5 天的极短窗口)
        # 这样能有效保证 Trigger Rate 控制在 5% - 15% 之间，既非连续变量也不至于极度低频
        signal = raw_signal.replace(0.0, np.nan).ffill(limit=4).fillna(0.0)
        
        signal.name = self.name
        return signal