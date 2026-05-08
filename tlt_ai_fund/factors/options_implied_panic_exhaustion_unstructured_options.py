import numpy as np
import pandas as pd

class OptionsImpliedPanicExhaustionFactor:
    """波动率期权利差衰竭因子 (unstructured/options)

    逻辑: 采用跨资产期权隐含波动率(VIX vs GVZ)的边际变动捕捉宏观恐慌的极值与衰竭。当 VIX(权益/通缩恐慌) 的边际增速远超 GVZ(黄金/通胀恐慌) 时，利差剧烈飙升，这往往是对极端通缩或流动性危机的定价，一旦动量衰竭，预示着美联储即将下场干预(看多美债)；反之则代表极端的通胀/信用超发恐慌见顶(看空美债)。
    数据: vixcls (标普期权隐含波动率), gvzcls (黄金ETF期权隐含波动率)
    触发: VIX-GVZ利差的5日变动量 Z-Score > 2.5 (极值) 且 当日变动量低于前3日均值 (二阶衰竭)
    输出: 脉冲型信号，+1.0 (联储救市预期/通缩见顶)，-1.0 (滞胀恐慌见顶)，常态为 0.0
    """

    def __init__(self):
        self.name = 'options_implied_panic_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 信号，严格遵守零值休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 缺失字段处理
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算跨资产期权隐含波动率利差
        opt_spread = vix - gvz
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝对禁止使用波动率绝对水位，必须使用 5 日动量变化捕捉预期突变瞬间
        spread_diff = opt_spread.diff(5)
        
        # 计算 252 日滚动 Z-Score (体现长期视角的极端跳跃)
        roll_mean = spread_diff.rolling(252).mean()
        roll_std = spread_diff.rolling(252).std().replace(0, np.nan)
        z_score = (spread_diff - roll_mean) / roll_std
        
        # 铁律2: 二阶导数衰竭 (Anti-Catch-Falling-Knife)
        # 禁止在单边暴涨暴跌中接飞刀，必须等待动能二阶导出现拐点
        diff_3d_mean = spread_diff.rolling(3).mean()
        exhaustion_long = spread_diff < diff_3d_mean  # 冲高回落
        exhaustion_short = spread_diff > diff_3d_mean # 探底回升
        
        # 铁律1: 狙击手脉冲 (Sniper Pulse)
        # 仅在同时满足“达到极值(>2.5)”且“出现衰竭”的瞬间触发 +/- 1.0
        long_trigger = ((z_score > 2.5) & exhaustion_long).fillna(False)
        short_trigger = ((z_score < -2.5) & exhaustion_short).fillna(False)
        
        signal[long_trigger] = 1.0
        signal[short_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"