import numpy as np
import pandas as pd

class CrossAssetVolatilityExhaustionFactor:
    """期权跨资产波动率恐慌衰竭脉冲 (unstructured/options)

    逻辑: 当股市隐含波动率(VIX)相对黄金隐含波动率(GVZ)出现罕见的极端边际飙升时，往往预示着经济衰退或流动性危机爆发的最高潮。
          根据二阶导数和零值休眠铁律，我们不接飞刀，而是等待这种极端跨资产相对恐慌的动量开始衰竭（见顶回落）时，
          往往能精准捕捉到美联储被迫转向救市的预期发酵点，此时长端美债迎来流动性修复或降息Price-in的主升浪。
          反之，若黄金波动率相对暴增并开始衰竭，则标志着恶性通胀恐慌高点确认，紧缩预期加剧，触发看空美债信号。
    数据: vixcls (标普500隐含波动率), gvzcls (黄金ETF隐含波动率)
    触发: 波动率剪刀差 5日变化量的 252日 Z-Score > 2.5 且 开始见顶回落 (二阶导 diff < 0)
    输出: 严格的狙击手脉冲信号 (+1.0 或 -1.0)，非触发日绝对休眠 (0.0)
    """

    def __init__(self):
        self.name = 'options_cross_vol_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化非触发日严格休眠为 0.0
        signal = pd.Series(0.0, index=data.index)
        signal.name = self.name
        
        # 数据完整性检查
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝对禁止直接使用绝对水位评估恐慌，只看"超额恐慌预期的突变"
        # spread_chg 代表了 1周内(5个交易日) 跨资产避险情绪的剧烈演化
        spread = vix - gvz
        spread_chg = spread.diff(5)
        
        # 提取历史统计基准，计算滚动 Z-Score (1年自然交易日基准)
        roll_mean = spread_chg.rolling(window=252, min_periods=60).mean()
        roll_std = spread_chg.rolling(window=252, min_periods=60).std()
        z_score = (spread_chg - roll_mean) / (roll_std + 1e-8)
        
        # 铁律2: 二阶导数衰竭 (Anti-Catch-Falling-Knife)
        # 必须等极端动量开始衰竭退潮，严防主跌浪中途买入
        chg_diff = spread_chg.diff(1)
        
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 极其严苛的条件同时共振才能触发信号：
        # 正向脉冲(+1.0): 股市相对恐慌极强飙升 (Z > 2.5) + 随后确认衰退 (chg_diff < 0) -> 政策宽松预期看多TLT
        cond_long = (z_score > 2.5) & (chg_diff < 0)
        
        # 反向脉冲(-1.0): 黄金相对恐慌极强飙升 (Z < -2.5) + 随后确认衰退 (chg_diff > 0) -> 滞胀紧缩预期看空TLT
        cond_short = (z_score < -2.5) & (chg_diff > 0)
        
        signal.loc[cond_long] = 1.0
        signal.loc[cond_short] = -1.0
        
        # 清理异常值和缺失点确保安全性
        signal = signal.fillna(0.0)
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"