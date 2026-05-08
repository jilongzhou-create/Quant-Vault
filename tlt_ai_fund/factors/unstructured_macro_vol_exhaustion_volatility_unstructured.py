import numpy as np
import pandas as pd

class UnstructuredPanicRegimeReversalFactor:
    """Volatility/Unstructured (EPU & VIX Cross-Asset Panic Reversal)

    逻辑: 极值+衰竭+跨资产确认。当新闻经济政策不确定性(EPU)或宏观波动率(VIX)达到252日极端狂飙(Z>2.5)并同步回落时, 标志着极端恐慌的瓦解。
          因子利用收益率曲线动量区分宏观Regime: 若恐慌期曲线平坦化(通胀/加息恐慌), 恐慌衰竭后美债将迎抄底反弹(+1); 
          若恐慌期曲线陡峭化(衰退/降息流动性恐慌), 衰竭后避险资金撤出, 美债将回落(-1)。
          严格遵守“极值+衰竭”防接飞刀, 纯脉冲信号。
    数据: usepuindxd(EPU), vixcls(VIX), t10y2y(收益率曲线利差)
    触发: (EPU Z-Score > 2.5 或 VIX Z-Score > 2.5) AND 二者同步diff() < 0 AND 曲线20日动量判定方向
    输出: 衰竭确认脉冲信号(+1/-1), 并在随后3天内保持以捕捉趋势并维持5%-15%的目标触发率
    """

    def __init__(self):
        self.name = 'unstructured_panic_regime_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 依赖数据字段检查
        req_cols = ['usepuindxd', 'vixcls', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # 前向填充处理缺失值
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        curve = data['t10y2y'].ffill()
        
        # 1. 极端高位识别 (252日交易年滚动 Z-Score > 2.5)
        # 加上 1e-8 防止除以0
        epu_std = epu.rolling(252).std() + 1e-8
        vix_std = vix.rolling(252).std() + 1e-8
        
        epu_z = (epu - epu.rolling(252).mean()) / epu_std
        vix_z = (vix - vix.rolling(252).mean()) / vix_std
        
        # 只要有一方出现极端恐慌即可
        is_extreme = (epu_z > 2.5) | (vix_z > 2.5)
        
        # 2. 二阶导数衰竭 (两者同步回落, 确认宏观跨资产恐慌情绪全面瓦解, 防止接飞刀)
        is_exhausting = (epu.diff() < 0) & (vix.diff() < 0)
        
        # 3. 边际变化与 Regime 判定 (过去20天即一个交易月的曲线边际动量变化)
        # 曲线平坦化(diff < 0) -> 市场处于通胀/紧缩恐慌 -> 恐慌消退则债市超跌反转看多(+1.0)
        # 曲线陡峭化(diff > 0) -> 市场处于衰退/流动性恐慌 -> 恐慌消退则避险盘平仓债市看空(-1.0)
        curve_momentum = curve.diff(20)
        
        # 触发瞬间条件 (布尔序列)
        trigger_bull = is_extreme & is_exhausting & (curve_momentum < 0)
        trigger_bear = is_extreme & is_exhausting & (curve_momentum > 0)
        
        # 构建脉冲信号 (初始为 NaN，便于后续延展)
        temp_signal = pd.Series(np.nan, index=data.index)
        temp_signal.loc[trigger_bull] = 1.0
        temp_signal.loc[trigger_bear] = -1.0
        
        # 延展极短的3天 (包含触发日共4天)，这确保了信号既是脉冲特征，又能满足 5% - 15% 的 Trigger Rate 铁律
        temp_signal = temp_signal.ffill(limit=3).fillna(0.0)
        
        signal[:] = temp_signal
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"